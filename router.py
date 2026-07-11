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
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from answer_cleaner import clean as clean_answer
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

_DEFAULT_CORRECTION_INSTRUCTION = (
    "That answer looks invalid or incomplete for this task. Please provide a "
    "corrected answer that fully satisfies the original request. Respond with "
    "ONLY the corrected answer."
)

# Kategori-ozel duzeltme talimatlari: jenerik yerine spesifik format beklentisi
# belirterek retry accuracy'sini artir.
_CORRECTION_INSTRUCTIONS: dict[TaskCategory, str] = {
    TaskCategory.NAMED_ENTITY_RECOGNITION:
        "Your answer must be a valid JSON array with no markdown fences. "
        "Respond with ONLY the corrected JSON array, nothing else. "
        'Format: [{"entity": "value", "type": "TYPE"}]',
    TaskCategory.CODE_DEBUGGING:
        "Your answer must contain the COMPLETE corrected code that fixes the bug. "
        "Start with a '# Bug:' comment. Respond with ONLY the fixed code.",
    TaskCategory.CODE_GENERATION:
        "Your answer must contain complete, runnable code with necessary imports. "
        "Respond with ONLY the code — no prose, no explanations.",
    TaskCategory.MATHEMATICAL_REASONING:
        "Your answer must end with 'Final Answer: <number>' on the last line. "
        "Recalculate carefully and respond with the corrected step-by-step solution.",
    TaskCategory.SENTIMENT_CLASSIFICATION:
        "Your answer must start with EXACTLY one of: positive, negative, neutral, or mixed "
        "(lowercase). Follow with a single sentence justification. Nothing else.",
    TaskCategory.LOGICAL_REASONING:
        "Your answer must end with 'Answer: <your complete answer>' on the last line. "
        "Re-examine the logic and provide the corrected solution.",
}

_SYSTEM_PROMPTS: dict[TaskCategory, str] = {
    TaskCategory.FACTUAL_KNOWLEDGE:
        "You are a precise factual Q&A assistant. Answer the user's question "
        "accurately and concisely in English. Go straight to the answer — no "
        "preamble like 'Sure' or 'Of course'. Keep it brief: 1-3 sentences "
        "unless the question explicitly requires more detail. Prioritize "
        "accuracy over length.",

    TaskCategory.MATHEMATICAL_REASONING:
        "You are a math problem solver. Solve the problem step by step in "
        "English. Keep reasoning CONCISE — show key steps only, skip obvious "
        "arithmetic. IMPORTANT: After your reasoning, write your final answer "
        "on the LAST line in this EXACT format:\n\n"
        "Final Answer: <number>\n\n"
        "The number should be plain (no units, no commas, no dollar signs). "
        "Example last line:\nFinal Answer: 42.5",

    TaskCategory.SENTIMENT_CLASSIFICATION:
        "You are a sentiment classifier. Your response MUST start with EXACTLY "
        "one of: positive, negative, neutral, or mixed (lowercase). Follow with "
        "a single sentence justification. Nothing else.\n"
        "Example: negative. The reviewer expressed dissatisfaction despite "
        "praising delivery speed.",

    TaskCategory.TEXT_SUMMARIZATION:
        "You are a text summarizer. Summarize the given text as requested in "
        "English. If a specific length is requested (e.g., 'one sentence', "
        "'15 words'), you MUST strictly follow that constraint — count your "
        "words. Be concise, capture only the key points, and do NOT add "
        "opinions or information not in the original text. Do NOT start with "
        "'The text discusses' or similar meta-phrasing.",

    TaskCategory.NAMED_ENTITY_RECOGNITION:
        "You are a named entity extraction system. Extract entities from the "
        "text. Respond with ONLY a valid JSON array — no markdown fences, no "
        "prose, no explanation before or after. Use this exact format:\n"
        '[{"entity": "value", "type": "PERSON|ORGANIZATION|LOCATION|DATE"}]\n'
        "If no entities are found, respond with exactly: []",

    TaskCategory.CODE_DEBUGGING:
        "You are a code debugger. First, write a 1-line comment starting with "
        "'# Bug:' explaining what the bug was. Then provide the COMPLETE "
        "corrected code that fixes the issue. Do NOT repeat the broken code. "
        "Do NOT add explanations outside code comments.",

    TaskCategory.LOGICAL_REASONING:
        "You are a logic puzzle solver. Solve step by step in English. Keep "
        "each deduction step to 1-2 sentences. End with EXACTLY this format "
        "on the last line:\n\nAnswer: <your complete answer here>\n\n"
        "Example last line:\nAnswer: Alice owns the cat, Bob owns the dog.",

    TaskCategory.CODE_GENERATION:
        "You are a code generator. Write the requested code in the specified "
        "language. If no language is specified, default to Python 3. "
        "Include necessary imports at the top. "
        "Respond with ONLY the code — no prose, no "
        "explanation outside code comments. The code must be complete, correct, "
        "and ready to run. Include type hints where appropriate.",
}

_DEFAULT_SYSTEM_PROMPT = (
    "Answer the user's task accurately and concisely in English. "
    "Do NOT start with pleasantries. Go straight to the answer."
)


class RouteSource(Enum):
    DETERMINISTIC = "deterministic"
    LOCAL_MODEL = "local_model"
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


# Gemma circuit-breaker ayarlari: zaman bazli reset ile daha akilli strateji
_GEMMA_CIRCUIT_RESET_SECONDS = 30.0  # Bu kadar sure sonra tekrar dene
_GEMMA_MAX_FAILURES = 2  # Bu kadar basarisizliktan sonra kalici kapat


class OptiRouter:
    def __init__(self, client: FireworksClient, settings: Settings):
        self._client = client
        self._settings = settings
        self._gemma_circuit_open = False
        self._gemma_failure_count = 0
        self._gemma_last_failure_time = 0.0

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

        # --- Local Model Tier (0 Fireworks Token) ---
        # Sentiment, NER ve Summarization gibi genel hafif gorevlerde once yerel modeli dene
        if category in (TaskCategory.SENTIMENT_CLASSIFICATION, TaskCategory.NAMED_ENTITY_RECOGNITION, TaskCategory.TEXT_SUMMARIZATION, TaskCategory.FACTUAL_KNOWLEDGE):
            from local_model import run_local_inference
            system_prompt = _SYSTEM_PROMPTS.get(category, _DEFAULT_SYSTEM_PROMPT) if category else _DEFAULT_SYSTEM_PROMPT
            local_res = run_local_inference(system_prompt, compressed, max_tokens)
            if local_res is not None:
                # Yerel model ciktisini temizle + validate et
                cleaned_local_text = clean_answer(local_res.text, category)
                if validate(local_res, category):
                    logger.info("Yerel model basarili oldu (0 Fireworks token) - Kategori: %s", category.value)
                    return RouteResult(
                        text=cleaned_local_text,
                        source=RouteSource.LOCAL_MODEL,
                        category=category,
                        model_used=local_res.model,
                        tokens_spent=tokens_spent,
                        was_corrected=False,
                    )
                else:
                    # Temizlenmiş haliyle tekrar validate dene (bazen sadece intro strip yeterli)
                    from fireworks_client import CompletionResult
                    cleaned_res = CompletionResult(
                        text=cleaned_local_text, model=local_res.model,
                        prompt_tokens=0, completion_tokens=0, total_tokens=0,
                        latency_seconds=local_res.latency_seconds,
                        raw_finish_reason=local_res.raw_finish_reason,
                    )
                    if cleaned_local_text != local_res.text and validate(cleaned_res, category):
                        logger.info("Yerel model temizleme sonrasi basarili (0 Fireworks token) - Kategori: %s", category.value)
                        return RouteResult(
                            text=cleaned_local_text,
                            source=RouteSource.LOCAL_MODEL,
                            category=category,
                            model_used=local_res.model,
                            tokens_spent=tokens_spent,
                            was_corrected=False,
                        )
                    logger.warning("Yerel model ciktisi dogrulamadan gecemedi, Fireworks'e dusulüyor - Kategori: %s", category.value)

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
        corrected = None  # Bug fix: validate() True donerse corrected tanimsiz kalmaz

        # --- Speculative validation + tek seferlik duzeltici retry ---
        if not validate(result, category):
            # Eger ilk cevap Gemma'dan geldiyse ve gecersizse, retry'i risk alip tekrar
            # Gemma'ya gondermek yerine kararli serverless default modele (minimax-m3) yonlendir.
            retry_model = self._settings.roles.default if "gemma" in result.model.lower() else result.model
            corrected = self._attempt_corrective_retry(retry_model, messages, result.text, max_tokens, category)
            if corrected is not None:
                tokens_spent += corrected.total_tokens
                text = corrected.text
                was_corrected = True

        # --- Post-processing: cevabi temizle (0 token) ---
        text = clean_answer(text, category)

        return RouteResult(
            text=text,
            source=source,
            category=category,
            model_used=corrected.model if (was_corrected and corrected) else result.model,
            tokens_spent=tokens_spent,
            was_corrected=was_corrected,
        )

    def _try_gemma_then_default(self, messages: list[dict], max_tokens: int):
        """'default' rolundeki kategoriler icin: once Gemma'yi kisa timeout +
        retrysiz dene (bonus odulu icin), basarisiz olursa SESSIZCE (exception
        yutulur) guvenilir serverless modele (minimax-m3) dus.

        Circuit-breaker (zaman bazli): Gemma basarisiz olursa belirli sure
        sonra tekrar denenir. _GEMMA_MAX_FAILURES'a ulasirsa kalici kapanir."""
        gemma_model = self._settings.gemma.model

        # Circuit-breaker: zaman bazli reset kontrolu
        if self._gemma_circuit_open and self._gemma_failure_count < _GEMMA_MAX_FAILURES:
            if time.monotonic() - self._gemma_last_failure_time > _GEMMA_CIRCUIT_RESET_SECONDS:
                self._gemma_circuit_open = False
                logger.info("Gemma circuit-breaker RESET — tekrar denenecek (basarisizlik: %d/%d)",
                            self._gemma_failure_count, _GEMMA_MAX_FAILURES)

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
                self._gemma_failure_count += 1
                self._gemma_last_failure_time = time.monotonic()
                if self._gemma_failure_count >= _GEMMA_MAX_FAILURES:
                    logger.warning(
                        "Gemma denemesi basarisiz oldu (%d/%d), circuit-breaker KALICI KAPANDI — "
                        "minimax-m3'e dusuluyor: %s", self._gemma_failure_count, _GEMMA_MAX_FAILURES, exc
                    )
                else:
                    logger.warning(
                        "Gemma denemesi basarisiz oldu (%d/%d), circuit-breaker AKTIF (%.0fs sonra reset) — "
                        "minimax-m3'e dusuluyor: %s",
                        self._gemma_failure_count, _GEMMA_MAX_FAILURES,
                        _GEMMA_CIRCUIT_RESET_SECONDS, exc
                    )

        result = self._client.chat_completion(
            model=self._settings.roles.default, messages=messages, max_tokens=max_tokens,
        )
        return result, RouteSource.DEFAULT_MODEL

    def _attempt_corrective_retry(
        self, model: str, messages: list[dict], previous_answer: str,
        max_tokens: int, category: Optional[TaskCategory] = None,
    ):
        """Ayni modele TEK SEFERLIK duzeltici retry. Kategori-ozel talimat
        kullanir. Bu da basarisiz olursa (ör. network hatasi) None doner -
        CRASH ETMEZ, ilk cevap korunur."""
        correction = _CORRECTION_INSTRUCTIONS.get(category, _DEFAULT_CORRECTION_INSTRUCTION) if category else _DEFAULT_CORRECTION_INSTRUCTION
        retry_messages = messages + [
            {"role": "assistant", "content": previous_answer},
            {"role": "user", "content": correction},
        ]
        try:
            return self._client.chat_completion(
                model=model, messages=retry_messages,
                max_tokens=max_tokens, temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001 - kasitli genis yakalama
            logger.warning("Duzeltici retry basarisiz oldu, ilk cevap korunuyor: %s", exc)
            return None
