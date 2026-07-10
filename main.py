"""
main.py
=======
Adaptive Model Dispatcher - Track 1 harness giris noktasi.

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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gorev listesini PARALEL olarak cozer (ThreadPoolExecutor).

    Zaman butcesi (deadline) asilirsa, kalan gorevlere YENI model cagrisi
    YAPILMAZ - hepsine bos cevap yazilir. Boylece:
      (a) /output/results.json her zaman HER task_id icin bir giris icerir,
      (b) 10 dakikalik sert sinira asla carpip sifir puanla sonuclanmayiz.

    Paralel islem: max_workers kadar gorev ayni anda cozulur. Bu, ozellikle
    Fireworks API cagrilarinin I/O-bagli oldugu durumlarda toplam sureyi
    onemli olcude azaltir. Thread-safety: UsageTracker lock'lu, her gorev
    izole try/except icerisinde.

    Doner: (results, task_details) - results harness icin, task_details dashboard icin.
    """
    total = len(tasks)
    max_workers = int(os.getenv("PARALLEL_WORKERS", "3"))

    def _solve_single(idx: int, task: dict) -> tuple[dict[str, Any], dict[str, Any]]:
        task_id = task["task_id"]
        prompt = task["prompt"]
        task_start = time.monotonic()

        if time.monotonic() >= deadline:
            logger.warning(
                "Zaman butcesi doldu - '%s' bos cevapla gecildi.", task_id,
            )
            detail = {
                "task_id": task_id, "category": None, "source": "timeout",
                "model_used": None, "tokens_spent": 0,
                "latency_seconds": 0.0, "was_corrected": False,
                "prompt_preview": prompt[:80],
            }
            return {"task_id": task_id, "answer": ""}, detail

        try:
            route_result = router.solve(prompt)
            latency = time.monotonic() - task_start
            logger.info(
                "[%d/%d] %s cozuldu: kaynak=%s kategori=%s model=%s token=%d duzeltildi=%s",
                idx, total, task_id, route_result.source.value,
                route_result.category.value if route_result.category else "-",
                route_result.model_used or "-", route_result.tokens_spent,
                route_result.was_corrected,
            )
            detail = {
                "task_id": task_id,
                "category": route_result.category.value if route_result.category else None,
                "source": route_result.source.value,
                "model_used": route_result.model_used,
                "tokens_spent": route_result.tokens_spent,
                "latency_seconds": round(latency, 3),
                "was_corrected": route_result.was_corrected,
                "prompt_preview": prompt[:80],
            }
            return {"task_id": task_id, "answer": route_result.text}, detail
        except DisallowedModelError:
            logger.exception(
                "[%d/%d] %s: izin verilmeyen model guard'i tetiklendi.", idx, total, task_id
            )
            detail = {
                "task_id": task_id, "category": None, "source": "error_disallowed",
                "model_used": None, "tokens_spent": 0,
                "latency_seconds": round(time.monotonic() - task_start, 3),
                "was_corrected": False, "prompt_preview": prompt[:80],
            }
            return {"task_id": task_id, "answer": ""}, detail
        except Exception:
            logger.exception(
                "[%d/%d] %s cozulurken beklenmeyen hata olustu, bos cevap yazildi.",
                idx, total, task_id,
            )
            detail = {
                "task_id": task_id, "category": None, "source": "error",
                "model_used": None, "tokens_spent": 0,
                "latency_seconds": round(time.monotonic() - task_start, 3),
                "was_corrected": False, "prompt_preview": prompt[:80],
            }
            return {"task_id": task_id, "answer": ""}, detail

    # Paralel calistirma: gorevler paralel cozulur, sirasi korunur
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # task_id sirasini korumak icin index bazli sonuc toplama
    results_by_idx: dict[int, dict[str, Any]] = {}
    details_by_idx: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_solve_single, idx, task): idx
            for idx, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, detail = future.result()
                results_by_idx[idx] = result
                details_by_idx[idx] = detail
            except Exception:
                # Bu noktaya asla gelmemeli (_solve_single her seyi yakaliyor)
                # ama guvenlik icin:
                task_id = tasks[idx - 1]["task_id"]
                logger.exception(
                    "[%d/%d] %s: future'dan beklenmeyen hata.", idx, total, task_id
                )
                results_by_idx[idx] = {"task_id": task_id, "answer": ""}
                details_by_idx[idx] = {
                    "task_id": task_id, "category": None, "source": "error_future",
                    "model_used": None, "tokens_spent": 0,
                    "latency_seconds": 0.0, "was_corrected": False,
                    "prompt_preview": tasks[idx - 1]["prompt"][:80],
                }

    # Sirali sonuc listesi olustur (task_id sirasina gore)
    results = [results_by_idx[i] for i in range(1, total + 1)]
    details = [details_by_idx[i] for i in range(1, total + 1)]
    return results, details



def generate_run_report(
    task_details: list[dict[str, Any]],
    usage_tracker,
    total_duration: float,
) -> dict[str, Any]:
    """Dashboard icin detayli calisma raporu olusturur."""
    from datetime import datetime, timezone

    # Model bazinda kullanim istatistikleri
    model_usage: dict[str, dict[str, int]] = {}
    for detail in task_details:
        model = detail.get("model_used") or "none"
        if model not in model_usage:
            model_usage[model] = {"calls": 0, "tokens": 0}
        model_usage[model]["calls"] += 1
        model_usage[model]["tokens"] += detail.get("tokens_spent", 0)

    # Kategori bazinda istatistikler
    category_stats: dict[str, dict[str, Any]] = {}
    for detail in task_details:
        cat = detail.get("category") or "unknown"
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "total_tokens": 0, "avg_latency": 0.0, "latencies": []}
        category_stats[cat]["count"] += 1
        category_stats[cat]["total_tokens"] += detail.get("tokens_spent", 0)
        category_stats[cat]["latencies"].append(detail.get("latency_seconds", 0.0))

    # Ortalama latency hesapla ve gecici listeyi temizle
    for cat, stats in category_stats.items():
        lats = stats.pop("latencies")
        stats["avg_latency"] = round(sum(lats) / len(lats), 3) if lats else 0.0

    total_tokens = sum(d.get("tokens_spent", 0) for d in task_details)
    corrected_count = sum(1 for d in task_details if d.get("was_corrected"))

    return {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(task_details),
        "total_tokens": total_tokens,
        "total_duration_seconds": round(total_duration, 2),
        "corrected_count": corrected_count,
        "tasks": task_details,
        "model_usage": model_usage,
        "category_stats": category_stats,
        "usage_tracker": {
            "call_count": usage_tracker.call_count,
            "prompt_tokens": usage_tracker.prompt_tokens,
            "completion_tokens": usage_tracker.completion_tokens,
            "total_tokens": usage_tracker.total_tokens,
        },
    }


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

    results, task_details = solve_all_tasks(tasks, router, deadline)

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

    # Dashboard icin run_report.json yaz (results.json ile ayni dizine)
    try:
        report = generate_run_report(task_details, client.usage, elapsed)
        report_path = settings.results_output_path.replace("results.json", "run_report.json")
        if report_path == settings.results_output_path:
            # Yol degismemisse (orn. custom dosya adi) yanina yaz
            report_path = settings.results_output_path + ".report.json"
        write_results(report_path, report)
        logger.info("Run report yazildi: %s", report_path)
    except Exception:
        # Rapor yazimi basarisiz olsa bile results.json zaten yazildi,
        # skor etkilenmez - sadece dashboard verisi eksik kalir.
        logger.warning("Run report yazilamadi (skor etkilenmez).", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

