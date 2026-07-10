"""
config.py
=========
Adaptive Model Dispatcher icin merkezi konfigurasyon modulu.

Sorumluluklari:
  1. .env dosyasini / ortam degiskenlerini oku.
  2. Yarismanin ZORUNLU tuttugu FIREWORKS_BASE_URL ve ALLOWED_MODELS
     degerlerini validate et.
  3. ALLOWED_MODELS icindeki 5 resmi Track 1 modelini (minimax-m3,
     kimi-k2p7-code, ve 3 Gemma 4 varyanti) rollere (default/reasoning/code)
     ayiran, gercek fiyat/serverless verisine dayali bir katalog sun.
  4. Gemma 4 On-Demand Deployment ayarlarini (bonus odulu icin) tut.

NOT: MODEL_CATALOG asagida 2026-07-08 itibariyle Fireworks AI uzerinde
dogrulanmis gercek verilerle dolduruldu (bkz. modul ici yorumlar). ALLOWED_MODELS
icinde katalogda olmayan YENI bir model gorulurse, otomatik olarak parametre
sayisi heuristic'ine (isimdeki 'Nb' deseni) duser - ama en guvenlisi boyle bir
durumda LIGHT/HEAVY yerine gecen DEFAULT_MODEL/REASONING_MODEL/CODE_MODEL
degiskenlerini .env icinde elle sabitlemek.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # .env varsa yukle; yoksa sessizce gecer (prod ortaminda env dogrudan gelir)


class ConfigError(RuntimeError):
    """Konfigurasyon eksik/hatali oldugunda firlatilir. Uygulamayi erken ve
    net bir hata mesajiyla durdurmak, yarisma sirasinda sessiz basarisizliktan
    (ve bosuna harcanan token'lardan) cok daha iyidir."""


def _parse_allowed_models(raw: str) -> list[str]:
    """'model_a, model_b,model_c' -> ['model_a', 'model_b', 'model_c']"""
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


# ==============================================================================
# MODEL_CATALOG: Track 1'in resmi ALLOWED_MODELS listesindeki 5 model icin
# dogrulanmis statik metadata (fiyat, serverless durumu, hangi gorev
# kategorilerine uygun oldugu). Kaynak: Fireworks AI model sayfalari +
# resmi fiyatlandirma (2026-07-08 itibariyle arastirildi).
#
# role_hints aciklamasi (Track 1'in 8 gorev kategorisiyle eslesir):
#   "default"   -> factual Q&A, sentiment, summarization, NER (genel, hafif)
#   "reasoning" -> math reasoning, logic puzzles (daha derin akil yurutme)
#   "code"      -> code debugging, code generation
#   "gemma_bonus" -> Gemma ailesi (bonus odulu takibi icin ayrica isaretlenir)
# ==============================================================================
MODEL_CATALOG: dict[str, dict] = {
    "accounts/fireworks/models/minimax-m3": {
        "role_hints": ("default", "reasoning"),
        "serverless": True,
        "provider": "minimax",
        "input_price_per_million": 0.30,
        "output_price_per_million": 1.20,
        "notes": "428B/23B-aktif MoE. Ucuz+guclu+serverless. Acik-agirlik indeksinde ust siralarda.",
    },
    "accounts/fireworks/models/kimi-k2p7-code": {
        "role_hints": ("code",),
        "serverless": True,
        "provider": "moonshot",
        "input_price_per_million": 0.95,
        "output_price_per_million": 4.00,
        "notes": "1T/32B-aktif MoE, kod-ozel. 'Always thinking' - basit gorevlerde bile "
                 "reasoning token'i harcar, SADECE code kategorilerinde kullan.",
    },
    "accounts/fireworks/models/gemma-4-31b-it": {
        "role_hints": ("default", "gemma_bonus"),
        "serverless": False,
        "provider": "google",
        "input_price_per_million": None,  # on-demand: GPU-saniye bazli, token fiyati yok
        "output_price_per_million": None,
        "notes": "31B dense. SADECE On-Demand Deployment (bkz. deployment_manager.py).",
    },
    "accounts/fireworks/models/gemma-4-26b-a4b-it": {
        "role_hints": ("default", "gemma_bonus"),
        "serverless": False,
        "provider": "google",
        "input_price_per_million": None,
        "output_price_per_million": None,
        "notes": "25.2B toplam/3.8B aktif MoE. SADECE On-Demand Deployment. "
                 "Az aktif parametre => genelde en hizli/ucuz Gemma deployment secenegi.",
    },
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4": {
        "role_hints": ("default", "gemma_bonus"),
        "serverless": False,
        "provider": "google",
        "input_price_per_million": None,
        "output_price_per_million": None,
        "notes": "31B, NVFP4 quantized. SADECE On-Demand Deployment.",
    },
}


def _estimate_param_count_billion(model_path: str) -> float:
    """MODEL_CATALOG'da olmayan (yarisma sonradan ekleyebilecegi) modeller icin
    yedek heuristic: isimdeki 'Nb' desenine bakar. Eslesme yoksa +inf doner
    (bilinmeyen modeli guvenli tarafta - 'buyuk/pahali' - tutmak icin)."""
    match = re.search(r"(\d+(?:\.\d+)?)b", model_path.lower())
    return float(match.group(1)) if match else float("inf")


@dataclass
class ModelRoles:
    """ALLOWED_MODELS listesinden turetilmis rol -> model path esleme."""

    default: str      # factual QA, sentiment, summarization, NER
    reasoning: str     # math reasoning, logic puzzles
    code: str          # code debugging, code generation
    all_models: list[str] = field(default_factory=list)
    gemma_models: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"  DEFAULT (genel)     : {self.default}",
            f"  REASONING (mantik)  : {self.reasoning}",
            f"  CODE (kodlama)      : {self.code}",
        ]
        if self.gemma_models:
            lines.append(f"  GEMMA ailesi (bonus) : {self.gemma_models}")
        return "\n".join(lines)


@dataclass
class GemmaDeploymentSettings:
    """Gemma 4 On-Demand Deployment ayarlari.

    ONEMLI (Participant Guide sonrasi netlesen gercek): Skorlanan harness
    KENDI FIREWORKS_API_KEY'ini enjekte ediyor ("provided by the harness -
    use this key, not your own") - yani bizim kendi hesabimizda kurdugumuz
    bir deployment, skorlanan kosuda muhtemelen ERISILEBILIR DEGIL (farkli
    hesap). Ayrica "Maximum runtime: 10 minutes" kurali, skorlanan kosu
    SIRASINDA sifirdan deployment kurmayi zaten imkansiz kiliyor.

    Bu yuzden bu ayarlar VE manage_gemma_deployment.py SADECE yerel
    gelistirme/test icin (Guide'in onerdigi "run a local eval step" adimi
    icin) kullanilir - kendi $50 hackathon kredinle Gemma'nin cevap
    kalitesini/prompt'unu ayarlamana yarar. Skorlanan kosuda router,
    Gemma'yi ALLOWED_MODELS'teki DUZ model ID'siyle (deployment suffix'i
    olmadan) normal bir cagri gibi dener; harness tarafi zaten hazirsa
    calisir, degilse router sessizce serverless modele (minimax-m3) duser.
    """

    model: Optional[str]
    deployment_id: str
    min_replica_count: int
    max_replica_count: int
    scale_to_zero_window: str
    accelerator_type: Optional[str]
    query_model_id: Optional[str]

    @property
    def is_configured(self) -> bool:
        return bool(self.model)

    @property
    def is_ready_to_query(self) -> bool:
        """'manage_gemma_deployment.py up' calistirilip GEMMA_QUERY_MODEL_ID
        .env'e elle eklendiyse True doner - SADECE yerel testte anlamlidir."""
        return bool(self.query_model_id)


@dataclass
class Settings:
    fireworks_api_key: str
    fireworks_base_url: str
    fireworks_account_id: Optional[str]
    allowed_models: list[str]
    accuracy_threshold: float
    request_timeout_seconds: int
    default_max_tokens: int
    tasks_input_path: str
    results_output_path: str
    max_runtime_seconds: int
    runtime_safety_margin_seconds: int
    roles: ModelRoles
    gemma: GemmaDeploymentSettings

    def is_model_allowed(self, model_path: str) -> bool:
        """Kritik guard: Router, ALLOWED_MODELS disinda HICBIR modele
        cagri yapamamali (deployment-scoped Gemma model string'leri
        '#account/deployment' suffix'i tasidigi icin ayrica kontrol edilir)."""
        if model_path in self.allowed_models:
            return True
        # Deployment-scoped Gemma cagrilari: "accounts/.../models/gemma-4-...#acct/dep-id"
        base = model_path.split("#", 1)[0]
        return base in self.allowed_models


def _catalog_lookup(allowed_models: list[str], role: str) -> Optional[str]:
    """MODEL_CATALOG'da verilen role_hint'e sahip, ALLOWED_MODELS icindeki,
    SERVERLESS bir model arar (deployment gerektirmeyen guvenli varsayilan).
    En ucuz input fiyatina sahip olani tercih eder."""
    candidates = [
        m for m in allowed_models
        if m in MODEL_CATALOG
        and role in MODEL_CATALOG[m]["role_hints"]
        and MODEL_CATALOG[m]["serverless"]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda m: MODEL_CATALOG[m]["input_price_per_million"] or float("inf"))


def _build_model_roles(allowed_models: list[str]) -> ModelRoles:
    if not allowed_models:
        raise ConfigError(
            "ALLOWED_MODELS bos! .env dosyanda en az bir model path'i olmali.\n"
            "Ornek: ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code"
        )

    explicit = {
        "default": os.getenv("DEFAULT_MODEL", "").strip() or None,
        "reasoning": os.getenv("REASONING_MODEL", "").strip() or None,
        "code": os.getenv("CODE_MODEL", "").strip() or None,
    }
    for role, model in explicit.items():
        if model and model not in allowed_models:
            raise ConfigError(f"{role.upper()}_MODEL='{model}' ALLOWED_MODELS listesinde degil: {allowed_models}")

    resolved = {}
    for role in ("default", "reasoning", "code"):
        if explicit[role]:
            resolved[role] = explicit[role]
            continue
        catalog_pick = _catalog_lookup(allowed_models, role)
        if catalog_pick:
            resolved[role] = catalog_pick
        else:
            # Son care: parametre-sayisi heuristic'i (katalogda olmayan yeni modeller icin).
            ranked = sorted(allowed_models, key=_estimate_param_count_billion)
            resolved[role] = ranked[0] if role != "reasoning" else ranked[-1]

    gemma_models = [
        m for m in allowed_models
        if (m in MODEL_CATALOG and MODEL_CATALOG[m]["provider"] == "google") or "gemma" in m.lower()
    ]

    return ModelRoles(
        default=resolved["default"],
        reasoning=resolved["reasoning"],
        code=resolved["code"],
        all_models=allowed_models,
        gemma_models=gemma_models,
    )


def _build_gemma_settings(allowed_models: list[str]) -> GemmaDeploymentSettings:
    model = os.getenv("GEMMA_MODEL", "").strip() or None
    if model and model not in allowed_models:
        raise ConfigError(f"GEMMA_MODEL='{model}' ALLOWED_MODELS listesinde degil: {allowed_models}")

    return GemmaDeploymentSettings(
        model=model,
        deployment_id=os.getenv("GEMMA_DEPLOYMENT_ID", "optiroute-gemma").strip(),
        min_replica_count=int(os.getenv("GEMMA_MIN_REPLICA_COUNT", "0")),
        max_replica_count=int(os.getenv("GEMMA_MAX_REPLICA_COUNT", "1")),
        scale_to_zero_window=os.getenv("GEMMA_SCALE_TO_ZERO_WINDOW", "10m").strip(),
        accelerator_type=os.getenv("GEMMA_ACCELERATOR_TYPE", "").strip() or None,
        query_model_id=os.getenv("GEMMA_QUERY_MODEL_ID", "").strip() or None,
    )


def load_settings() -> Settings:
    """Tum .env / ortam degiskenlerini okuyup dogrulanmis bir Settings nesnesi doner.
    Eksik/gecersiz bir sey varsa ConfigError firlatir - uygulama en basta,
    ilk Fireworks cagrisindan ONCE cokmeli."""

    api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
    base_url = os.getenv("FIREWORKS_BASE_URL", "").strip()
    allowed_raw = os.getenv("ALLOWED_MODELS", "").strip()

    missing = []
    if not api_key:
        missing.append("FIREWORKS_API_KEY")
    if not base_url:
        missing.append("FIREWORKS_BASE_URL")
    if not allowed_raw:
        missing.append("ALLOWED_MODELS")
    if missing:
        raise ConfigError(
            "Zorunlu ortam degiskenleri eksik: " + ", ".join(missing) +
            "\n.env dosyani kontrol et (bkz. .env.example)."
        )

    allowed_models = _parse_allowed_models(allowed_raw)
    roles = _build_model_roles(allowed_models)
    gemma = _build_gemma_settings(allowed_models)

    return Settings(
        fireworks_api_key=api_key,
        fireworks_base_url=base_url,
        fireworks_account_id=os.getenv("FIREWORKS_ACCOUNT_ID", "").strip() or None,
        allowed_models=allowed_models,
        accuracy_threshold=float(os.getenv("ACCURACY_THRESHOLD", "0.8")),
        # Guide: "Response time per request must be under 30 seconds" - 25s
        # varsayilani, kendi isleme overhead'imiz icin birkac saniyelik pay birakir.
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "25")),
        default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "1024")),
        # Guide: harness /input/tasks.json'i okur, /output/results.json'a yazariz.
        # Bu yollar env ile override edilebilir (lokal testte gercek / dizinine
        # yazma izni olmayabilir) ama varsayilan DEGERLER harness'in beklediginin
        # AYNISI - .env'de hicbir sey ayarlanmasa bile dogru calisir.
        tasks_input_path=os.getenv("TASKS_INPUT_PATH", "/input/tasks.json").strip(),
        results_output_path=os.getenv("RESULTS_OUTPUT_PATH", "/output/results.json").strip(),
        # Guide: "Maximum runtime: 10 minutes". Varsayilan 600s; safety_margin
        # kadar erken kesilir ki /output/results.json'i yazmaya HER ZAMAN vaktimiz olsun.
        max_runtime_seconds=int(os.getenv("MAX_RUNTIME_SECONDS", "600")),
        runtime_safety_margin_seconds=int(os.getenv("RUNTIME_SAFETY_MARGIN_SECONDS", "20")),
        roles=roles,
        gemma=gemma,
    )


if __name__ == "__main__":
    # Hizli manuel dogrulama: `python config.py`
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("Konfigurasyon basariyla yuklendi.")
    print(f"  FIREWORKS_BASE_URL : {settings.fireworks_base_url}")
    print(f"  ALLOWED_MODELS     : {settings.allowed_models}")
    print(settings.roles.describe())
    print(f"  ACCURACY_THRESHOLD : {settings.accuracy_threshold}")
    print(f"  TASKS_INPUT_PATH   : {settings.tasks_input_path}")
    print(f"  RESULTS_OUTPUT_PATH: {settings.results_output_path}")
    print(f"  MAX_RUNTIME_SECONDS: {settings.max_runtime_seconds} (guvenlik payi: {settings.runtime_safety_margin_seconds}s)")
    print("\n  --- Gemma Deployment ---")
    print(f"  Yapilandirildi mi?  : {settings.gemma.is_configured}")
    print(f"  Sorguya hazir mi?   : {settings.gemma.is_ready_to_query}")
    if not settings.gemma.is_ready_to_query:
        print("  (Henuz hazir degil - 'python manage_gemma_deployment.py up' calistir)")
