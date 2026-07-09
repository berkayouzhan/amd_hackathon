"""
router.py
=========
OptiRouter - tum tier'lari (deterministic -> triage -> model -> validate ->
duzeltici retry) birlestiren orkestrator. Sistemin TEK karar noktasi budur;
main.py sadece bunu cagirir.

Akis (bkz. HANDOFF.md / README.md icin ayrintili gerekce):
  compress_prompt()
    -> Tier 0 (deterministic_solver) - eslesirse 0 token, BITTI
    -> triage.classify() - kategori belirle (0-token heuristic, belirsizse
       ucuz model cagrisi)
    -> "default" kategoriler: once Gemma (kisa timeout, retrysiz) dene ->
       basarisizsa (herhangi bir exception) SESSIZCE minimax-m3'e dus
    -> "reasoning"/"code" kategoriler: dogrudan minimax-m3 / kimi-k2p7-code
       (Gemma'ya HIC dokunulmaz)
    -> validator.validate() - supheliyse AYNI modele TEK SEFERLIK duzeltici
       retry (retry de basarisiz olursa - ornegin network hatasi - ilk
       cevap korunur, CRASH edilmez)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import Settings
from deterministic_solver import try_solve as try_deterministic_solve
from fireworks_client import FireworksClient
from prompt_compressor import compress_prompt
from triage import CATEGORY_PROFILES, ROLE_BY_CATEGORY, TaskCategory, classify
from validator import validate

logger = logging.getLogger("optiroute.router")

# Gemma bonus denemesi icin: kisa timeout + retrysiz (speculative). Gemma
# yavas/hazir degilse normal fallback cagrisini geciktirmemesi icin.
_GEMMA_ATTEMPT_TIMEOUT_SECONDS = 6.0

_CORRECTION_INSTRUCTION = (
    "That answer looks invalid or incomplete for this task. Please provide a "
    "corrected answer that fully satisfies the original request. Respond with "
    "ONLY the corrected answer."
)

_SYSTEM_PROMPTS: dict[TaskCategory, str] = {
    TaskCategory.FACTUAL_KNOWLEDGE:
        "Answer the user's factual question accurately and concisely, in English.",
    TaskCategory.MATHEMATICAL_REASONING:
        "Solve the math problem step by step, then give the final numeric answer "
        "clearly, in English.",
    TaskCategory.SENTIMENT_CLASSIFICATION:
        "Classify the sentiment of the given text (positive, negative, or neutral) "
        "and briefly justify your answer, in English.",
    TaskCategory.TEXT_SUMMARIZATION:
        "Summarize the given text as requested, in English.",
    TaskCategory.NAMED_ENTITY_RECOGNITION:
        "Extract the requested named entities and respond with ONLY valid JSON, "
        "no prose, no markdown fences.",
    TaskCategory.CODE_DEBUGGING:
        "Find and fix the bug in the given code. Respond with the corrected code.",
    TaskCategory.LOGICAL_REASONING:
        "Solve the logic puzzle step by step, then clearly state the final "
        "conclusion, in English.",
    TaskCategory.CODE_GENERATION:
        "Write the requested code. Respond with ONLY the code (plus minimal "
        "explanation if truly necessary).",
}

_DEFAULT_SYSTEM_PROMPT = "Answer the user's task accurately and concisely, in English."


class RouteSource(Enum):
    DETERMINISTIC = "deterministic"
    GEMMA_BONUS = "gemma_bonus"
    DEFAULT_MODEL = "default_model"
    REASONING_MODEL = "reasoning_model"
    CODE_MODEL = "code_model"


@dataclass
class RouteResult:
    text: str
    source: RouteSource
    category: Optional[TaskCategory]
    model_used: Optional[str]
    tokens_spent: int
    was_corrected: bool = False


def _build_messages(category: Optional[TaskCategory], prompt: str) -> list[dict]:
    system = _SYSTEM_PROMPTS.get(category, _DEFAULT_SYSTEM_PROMPT) if category else _DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


class OptiRouter:
    def __init__(self, client: FireworksClient, settings: Settings):
        self._client = client
        self._settings = settings
        self._gemma_circuit_open = False  # Ilk basarisizliktan sonra True olur

    def solve(self, prompt: str) -> RouteResult:
        compressed = compress_prompt(prompt)

        # --- Tier 0: deterministik cozucu (0 token) ---
        deterministic_answer = try_deterministic_solve(compressed)
        if deterministic_answer is not None:
            return RouteResult(
                text=deterministic_answer,
                source=RouteSource.DETERMINISTIC,
                category=None,
                model_used=None,
                tokens_spent=0,
                was_corrected=False,
            )

        # --- Triage: kategori belirle ---
        category, triage_tokens = classify(compressed, self._client, self._settings)
        role = ROLE_BY_CATEGORY[category]
        max_tokens = CATEGORY_PROFILES.get(category, self._settings.default_max_tokens)
        messages = _build_messages(category, compressed)

        tokens_spent = triage_tokens
        result = None
        source: Optional[RouteSource] = None

        if role == "default":
            result, source = self._try_gemma_then_default(messages, max_tokens)
        elif role == "reasoning":
            result = self._client.chat_completion(
                model=self._settings.roles.reasoning, messages=messages, max_tokens=max_tokens,
            )
            source = RouteSource.REASONING_MODEL
        else:  # "code"
            result = self._client.chat_completion(
                model=self._settings.roles.code, messages=messages, max_tokens=max_tokens,
            )
            source = RouteSource.CODE_MODEL

        tokens_spent += result.total_tokens
        text = result.text
        was_corrected = False

        # --- Speculative validation + tek seferlik duzeltici retry ---
        if not validate(result, category):
            corrected = self._attempt_corrective_retry(result.model, messages, result.text, max_tokens)
            if corrected is not None:
                tokens_spent += corrected.total_tokens
                text = corrected.text
                was_corrected = True

        return RouteResult(
            text=text,
            source=source,
            category=category,
            model_used=result.model,
            tokens_spent=tokens_spent,
            was_corrected=was_corrected,
        )

    def _try_gemma_then_default(self, messages: list[dict], max_tokens: int):
        """'default' rolundeki kategoriler icin: once Gemma'yi kisa timeout +
        retrysiz dene (bonus odulu icin), basarisiz olursa SESSIZCE (exception
        yutulur) guvenilir serverless modele (minimax-m3) dus.

        Circuit-breaker: Gemma bir kez basarisiz olursa, kalan gorevlerde
        tekrar denenmez - her gorev icin 6s timeout kaybini onler."""
        gemma_model = self._settings.gemma.model
        if gemma_model and not self._gemma_circuit_open:
            try:
                result = self._client.chat_completion(
                    model=gemma_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout_seconds=_GEMMA_ATTEMPT_TIMEOUT_SECONDS,
                    retryable=False,
                )
                return result, RouteSource.GEMMA_BONUS
            except Exception as exc:  # noqa: BLE001 - kasitli genis yakalama
                self._gemma_circuit_open = True
                logger.warning(
                    "Gemma denemesi basarisiz oldu, circuit-breaker AKTIF — "
                    "kalan gorevlerde Gemma denenmeyecek, minimax-m3'e dusuluyor: %s", exc
                )

        result = self._client.chat_completion(
            model=self._settings.roles.default, messages=messages, max_tokens=max_tokens,
        )
        return result, RouteSource.DEFAULT_MODEL

    def _attempt_corrective_retry(self, model: str, messages: list[dict], previous_answer: str, max_tokens: int):
        """Ayni modele TEK SEFERLIK duzeltici retry. Bu da basarisiz olursa
        (ör. network hatasi) None doner - CRASH ETMEZ, ilk cevap korunur."""
        retry_messages = messages + [
            {"role": "assistant", "content": previous_answer},
            {"role": "user", "content": _CORRECTION_INSTRUCTION},
        ]
        try:
            return self._client.chat_completion(model=model, messages=retry_messages, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 - kasitli genis yakalama
            logger.warning("Duzeltici retry basarisiz oldu, ilk cevap korunuyor: %s", exc)
            return None
