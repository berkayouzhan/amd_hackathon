"""
main.py
=======
OptiRoute AI - Track 1 harness giris noktasi.

PARTICIPANT GUIDE KONTRATI (birebir uygulanir):
  1. /input/tasks.json oku:
         [{"task_id": "t1", "prompt": "..."}, ...]
  2. Her gorevi coz.
  3. /output/results.json'a yaz (cikmadan ONCE):
         [{"task_id": "t1", "answer": "..."}, ...]
  4. Basarili -> exit code 0. Hatali -> non-zero.

SERT KURALLAR (Guide'dan, hepsi asagida ele alinir):
  - Toplam calisma suresi <= 10 dakika: kendi ic zaman butcemizi (deadline)
    yonetiriz, sure dolmaya yaklasinca YENI gorevlere baslamayi durdurup
    kalanlari bos cevapla doldururuz - boylece HER ZAMAN gecerli, TAM bir
    results.json yaziyoruz (malformed/eksik output = sifir puan; "hicbir
    sey yazmamak" en kotu senaryo, "bos cevap yazmak" en azindan gecerli).
  - Tek istek ~30 saniye (bkz. fireworks_client.py'deki retry/timeout ayari).
  - TUM cevaplar Ingilizce olmali (bkz. router.py sistem promptu).
  - Bir gorevin patlamasi TUM batch'i dusurmemeli - her gorev kendi
    try/except'i icinde izole calisir, hata durumunda bos cevapla devam edilir.
  - Sadece ALLOWED_MODELS uzerinden cagri (guard: fireworks_client.py).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from config import ConfigError, Settings, load_settings
from fireworks_client import DisallowedModelError, FireworksClient
from router import OptiRouter

# SADECE yerel test icin: USE_FAKE_FIREWORKS=1 ayarlanirsa, gercek Fireworks
# API'sine HICBIR cagri yapilmaz - onun yerine offline sahte cevaplar ureten
# FakeFireworksClient kullanilir (bkz. fake_fireworks_client.py). Boylece
# Fireworks kredisi/API key'i olmadan da main.py'nin TUM batch akisini
# (Tier 0 -> triage -> Gemma-once-dene -> validate -> retry -> I/O -> deadline)
# uctan uca gozlemleyebilirsin. Skorlanan kosuda bu env var ASLA set edilmemeli.
_USE_FAKE_FIREWORKS = os.getenv("USE_FAKE_FIREWORKS", "").strip().lower() in ("1", "true", "yes")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("optiroute.main")


def read_tasks(path: str) -> list[dict]:
    """/input/tasks.json'i okur ve validate eder. Guide'in sema
    ornegiyle birebir uyumlu: [{"task_id": "...", "prompt": "..."}, ...]"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gorev dosyasi bulunamadi: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} bir JSON listesi olmali, {type(data).__name__} bulundu.")
    for i, task in enumerate(data):
        if not isinstance(task, dict) or "task_id" not in task or "prompt" not in task:
            raise ValueError(f"Gorev #{i} 'task_id' veya 'prompt' alanini icermiyor: {task}")
    return data


def write_results(path: str, results: list[dict[str, Any]]) -> None:
    """Atomik yazma: once '<path>.tmp' dosyasina yaz, sonra os.replace() ile
    hedefe tasi. Boylece yazma sirasinda kesilirsek (ör. 10dk siniri veya
    beklenmedik bir crash) diskte YARIM/BOZUK bir results.json ASLA kalmaz -
    ya tam gecerli yeni dosya vardir ya da islem hic baslamamis gibi olur."""
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def solve_all_tasks(
    tasks: list[dict], router: OptiRouter, deadline: float
) -> list[dict[str, Any]]:
    """Gorev listesini SIRAYLA cozer.

    Zaman butcesi (deadline) asilirsa, kalan gorevlere YENI model cagrisi
    YAPILMAZ - hepsine bos cevap yazilir. Boylece:
      (a) /output/results.json her zaman HER task_id icin bir giris icerir
          (harness'in sonuclari task_id'ye gore eslestirdigini varsayarak,
          eksik giris birakmak "o gorev icin cevap yok" riskinden daha
          tehlikeli olabilir),
      (b) 10 dakikalik sert sinira asla carpip sifir puanla sonuclanmayiz.

    Tek bir gorevin (beklenmedik exception, izin verilmeyen model guard'i
    vb.) patlamasi asla tum batch'i dusurmez - her gorev izole calisir.
    """
    results: list[dict[str, Any]] = []
    total = len(tasks)

    for idx, task in enumerate(tasks, start=1):
        task_id = task["task_id"]
        prompt = task["prompt"]

        if time.monotonic() >= deadline:
            logger.warning(
                "Zaman butcesi doldu (%d/%d gorev islendi) - '%s' bos cevapla gecildi.",
                idx - 1, total, task_id,
            )
            results.append({"task_id": task_id, "answer": ""})
            continue

        try:
            route_result = router.solve(prompt)
            results.append({"task_id": task_id, "answer": route_result.text})
            logger.info(
                "[%d/%d] %s cozuldu: kaynak=%s kategori=%s model=%s token=%d duzeltildi=%s",
                idx, total, task_id, route_result.source.value,
                route_result.category.value if route_result.category else "-",
                route_result.model_used or "-", route_result.tokens_spent,
                route_result.was_corrected,
            )
        except DisallowedModelError:
            # Guard'in kendisi asla tetiklenmemeli (config zaten engelliyor)
            # ama tetiklenirse CRASH ETMEK yerine bos cevapla devam et -
            # tek gorevin hatasi tum submission'i riske atmamali.
            logger.exception(
                "[%d/%d] %s: izin verilmeyen model guard'i tetiklendi.", idx, total, task_id
            )
            results.append({"task_id": task_id, "answer": ""})
        except Exception:
            logger.exception(
                "[%d/%d] %s cozulurken beklenmeyen hata olustu, bos cevap yazildi.",
                idx, total, task_id,
            )
            results.append({"task_id": task_id, "answer": ""})

    return results


def main() -> int:
    run_start = time.monotonic()

    try:
        settings = load_settings()
    except ConfigError as e:
        logger.error("Konfigurasyon hatasi: %s", e)
        return 1

    logger.info("Ayarlar yuklendi:\n%s", settings.roles.describe())
    if settings.gemma.is_configured:
        # NOT: Bu SADECE yerel test icin anlamli - skorlanan kosuda router
        # Gemma'yi ALLOWED_MODELS'teki duz model ID'siyle dener (bkz. router.py).
        status = "yerel testte hazir" if settings.gemma.is_ready_to_query else "yerelde kurulmamis"
        logger.info("Gemma (yerel test) durumu: %s", status)

    try:
        tasks = read_tasks(settings.tasks_input_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error("Gorev dosyasi okunamadi (%s): %s", settings.tasks_input_path, e)
        return 1

    logger.info("%d gorev okundu: %s", len(tasks), settings.tasks_input_path)

    # Guide: "Maximum runtime: 10 minutes". Bu deadline'i asmamak icin
    # guvenlik payi kadar erken duruyoruz (results.json'i yazmaya HER ZAMAN
    # vaktimiz olsun diye).
    deadline = run_start + settings.max_runtime_seconds - settings.runtime_safety_margin_seconds

    if _USE_FAKE_FIREWORKS:
        from fake_fireworks_client import FakeFireworksClient
        logger.warning(
            "USE_FAKE_FIREWORKS=1 - SAHTE (offline) Fireworks client kullaniliyor. "
            "Gercek API'ye HICBIR cagri yapilmayacak. Bu SADECE yerel gelistirme "
            "icindir, skorlanan kosuda KESINLIKLE kullanilmamali."
        )
        client = FakeFireworksClient(settings)
    else:
        client = FireworksClient(settings)
    router = OptiRouter(client, settings)

    results = solve_all_tasks(tasks, router, deadline)

    try:
        write_results(settings.results_output_path, results)
    except OSError as e:
        logger.error("Sonuclar yazilamadi (%s): %s", settings.results_output_path, e)
        return 1

    elapsed = time.monotonic() - run_start
    logger.info(
        "Tamamlandi: %d/%d gorev, %.1fs surdu, sonuclar yazildi: %s",
        len(results), len(tasks), elapsed, settings.results_output_path,
    )
    logger.info("--- Kumulatif Kullanim Raporu ---\n%s", client.usage.report())

    return 0


if __name__ == "__main__":
    sys.exit(main())
