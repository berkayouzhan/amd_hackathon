# Adaptive Model Dispatcher — Proje Durumu / Devir Notu

> **2026-07-09 GÜNCELLEMESİ:** Yeni bir Claude oturumunda, kullanıcı sadece
> `config.py`, `fireworks_client.py`, `main.py`, `Dockerfile`, `tasks.json`
> dosyalarını yükledi — **`router.py` ve bağımlılıkları (`deterministic_solver.py`,
> `triage.py`, `prompt_compressor.py`, `validator.py`) hiç yüklenmemişti**
> (önceki oturumda yazılmış olabilirler ama bu container'a taşınmadılar).
> Bu dosyaların hepsi, bu HANDOFF'un aşağıdaki bölümlerinde anlatılan davranış
> sözleşmesine ve `tests/test_router.py`'nin beklediği kontratlara birebir uyacak
> şekilde SIFIRDAN yeniden yazıldı ve `pytest`'siz manuel bir test koşucusuyla
> (sandbox'ta network/pytest kurulumu yoktu) tüm 6 test doğrulandı — 6/6 PASS.
>
> Ayrıca kullanıcı "Fireworks kredisi henüz yok, onsuz devam edelim" dediği için
> YENİ bir dosya eklendi: **`fake_fireworks_client.py`** — `USE_FAKE_FIREWORKS=1`
> ile `main.py`'yi gerçek API'ye hiç dokunmadan uçtan uca (8/8 görev, tüm 8
> kategori doğru sınıflandı, Gemma-fallback ve düzeltici-retry yolları dahil)
> çalıştırabilmeyi sağlıyor. Bu SADECE yerel geliştirme için, skorlanan koşuda
> kullanılmamalı (main.py bunu env var yokken hiç import bile etmiyor).
>
> **Hâlâ eksik / bu oturumda dokunulmadı:** `deployment_manager.py`,
> `manage_gemma_deployment.py`, ve HANDOFF §3'te bahsedilen "52 test"in tamamı
> (bu oturumda sadece `test_router.py` yüklenmişti — diğer test dosyaları,
> örn. `test_deterministic_solver.py`, `test_triage.py`, `test_validator.py`,
> `test_prompt_compressor.py`, `test_config.py`, `test_main.py`, muhtemelen
> önceki oturumdaydı ama bu sefer yüklenmedi. Kullanıcı isterse onları da
> tekrar yüklemeli ya da yeniden yazdırmalı).

> Bu belge, projeyi başka bir Claude oturumuna (veya insana) devretmek için yazıldı.
> Amaç: sıfırdan bağlam kurmadan kaldığı yerden devam edebilmek.

## 1. Yarışma Bağlamı

**AMD Developer Hackathon: ACT II** — `lablab.ai` üzerinden yürütülüyor.
**Track 1: General-Purpose AI Agent** (siteye göre eski adı "Hybrid Token-Efficient
Routing Agent" idi, resmi Participant Guide'da "General-Purpose AI Agent" olarak geçiyor).

Kullanıcı bir veri mühendisi; **mimari yönlendirmeyi o yapıyor, tüm kodlamayı Claude
yapıyor**. Kullanıcı Türkçe konuşuyor, yanıtlar Türkçe olmalı.

### Resmi kurallar (Participant Guide PDF'inden — TEK otorite kaynağı, kullanıcı yükledi)

- **8 görev kategorisi** (ajan hepsini karşılamalı): factual knowledge, mathematical
  reasoning, sentiment classification, text summarization, named entity recognition,
  code debugging, logical/deductive reasoning, code generation.
- **I/O kontratı (kritik, tek-görev değil BATCH):**
  - Container **BİR KERE** çalışır, `/input/tasks.json`'daki **TÜM** görev listesini okur:
    `[{"task_id": "t1", "prompt": "..."}]`
  - Tüm sonuçları `/output/results.json`'a **çıkmadan önce** tek seferde yazar:
    `[{"task_id": "t1", "answer": "..."}]`
  - `results.json` bozuksa/eksikse → **sıfır puan**.
- **Ortam değişkenleri (harness enjekte eder):**
  `FIREWORKS_API_KEY` (harness'in kendi key'i, katılımcının kişisel key'i DEĞİL),
  `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` (virgülle ayrılmış).
- **Resmi ALLOWED_MODELS listesi (launch day'de Discord'da açıklandı):**
  `minimax-m3`, `kimi-k2p7-code`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`
- **Sert limitler:** Toplam çalışma süresi ≤10 dakika, istek başına ~30 saniye,
  container 60 saniye içinde hazır olmalı, image ≤10GB, tüm cevaplar İngilizce,
  hardcoded/cache cevap YASAK (unseen variants ile test ediliyor), linux/amd64
  manifest zorunlu.
- **Puanlama:** (1) LLM-Judge accuracy gate (eşiğin altı = leaderboard'dan tamamen
  elenir), (2) eşiği geçenler arasında toplam token'a göre artan sırada sıralanır
  (az token = yüksek sıra).
- **Gemma bonus:** Fireworks üzerinden Gemma 4'ün akıllıca kullanımı için ayrı bir
  ödül havuzu var (tam $ tutarı ve Track 1 payı teyit edilmedi).

### Araştırmayla düzeltilen kritik varsayımlar (ÖNEMLİ — tekrar düşmemek için)

1. Gemma 4'ün 3 varyantı da Fireworks'te SERVERLESS DEĞİL — sadece On-Demand
   Deployment (GPU-saniye bazlı ücret) ile çalışıyor.
2. AMA skorlanan koşuda harness kendi FIREWORKS_API_KEY'ini enjekte ediyor —
   katılımcının kişisel hesabında kurduğu bir deployment'a bu farklı hesap muhtemelen
   ERİŞEMEZ. Ayrıca 10 dakikalık sert süre sınırı, skorlanan koşu sırasında sıfırdan
   deployment kurmayı zaten imkansız kılıyor.
3. Sonuç/karar: Router, Gemma'yı ALLOWED_MODELS'teki düz model ID'siyle (deployment
   suffix'i olmadan) normal bir çağrı gibi dener; organizatörlerin tarafı zaten
   hazırsa çalışır, değilse sessizce serverless modele düşer. deployment_manager.py /
   manage_gemma_deployment.py araçları hâlâ var ama artık rolleri SADECE yerel
   geliştirme/test (katılımcının kendi $50 kredisiyle Gemma'nın çıktı kalitesini
   test etmesi için) — skorlanan yolun bir parçası değiller.

## 2. Mevcut Mimari (dosya dosya)

Proje kökü: `/mnt/user-data/outputs/optiroute-ai/` (ve çalışma kopyası
`/home/claude/optiroute-ai/`)

```
optiroute-ai/
├── config.py                  # env okuma + MODEL_CATALOG + rol ataması + Gemma ayarları
├── fireworks_client.py        # Fireworks API wrapper, ALLOWED_MODELS guard, UsageTracker
├── deterministic_solver.py    # Tier 0: 0-token çözücüler (konservatif)
├── triage.py                  # 8 kategoriye sınıflandırma (heuristic + model fallback)
├── prompt_compressor.py       # whitespace sıkıştırma (kod bloklarına dokunmaz)
├── validator.py               # speculative validation (0 token, deterministik kontroller)
├── router.py                  # OptiRouter — hepsini birleştiren orkestratör
├── main.py                    # harness giriş noktası (batch I/O kontratı)
├── deployment_manager.py      # Fireworks Deployment REST API wrapper (SADECE yerel test)
├── manage_gemma_deployment.py # CLI: up/status/down (SADECE yerel test)
├── Dockerfile                 # python:3.11-slim, root user, /input /output hazır
├── requirements.txt           # openai, python-dotenv, tenacity, requests
├── .env.example                # tüm env var'lar + açıklamaları
├── .gitignore
└── local_test/tasks.json      # 8 kategoriyi kapsayan örnek görev seti (yerel test için)
```

### config.py — Neyi Bilir
- MODEL_CATALOG: 5 resmi modelin doğrulanmış fiyat/serverless verisi (minimax-m3:
  $0.30/$1.20 per 1M, serverless; kimi-k2p7-code: $0.95/$4.00, serverless, "always
  thinking"; 3 Gemma varyantı: serverless YOK, on-demand only).
- ModelRoles: default/reasoning/code rollerini ALLOWED_MODELS'ten otomatik seçer
  (override: DEFAULT_MODEL/REASONING_MODEL/CODE_MODEL env). Şu an otomatik seçim:
  default+reasoning → minimax-m3, code → kimi-k2p7-code.
- GemmaDeploymentSettings: SADECE yerel test için (yukarıya bakın).
- Harness I/O yolları (TASKS_INPUT_PATH/RESULTS_OUTPUT_PATH, varsayılan
  /input/tasks.json / /output/results.json) ve zaman bütçesi
  (MAX_RUNTIME_SECONDS=600, RUNTIME_SAFETY_MARGIN_SECONDS=20).

### router.py — OptiRouter.solve() Akışı
```
compress_prompt()
  → Tier 0 (deterministic_solver) — eşleşirse 0 token, BİTTİ
  → triage.classify() — kategori belirle (0 token heuristic, belirsizse ucuz model)
  → "default" kategoriler: önce Gemma (6sn timeout, retrysiz) dene → başarısızsa minimax-m3
  → "reasoning"/"code" kategoriler: doğrudan minimax-m3 / kimi-k2p7-code
  → validator.validate() — şüpheliyse (boş/kesilmiş/ret/geçersiz JSON/sayı-yok/kod-yok)
    AYNI modele TEK SEFERLİK düzeltici retry (Track 1'de net bir "küçük→büyük"
    model merdiveni olmadığı için "daha güçlü modele yükselt" yerine bu tercih edildi)
```

### main.py — Harness Kontratı
- /input/tasks.json okur, HER görevi izole try/except içinde çözer (biri patlarsa
  diğerleri etkilenmez), zaman bütçesi dolunca yeni çağrı yapmadan boş cevapla doldurur,
  atomik yazma (*.tmp → os.replace) ile /output/results.json'ı asla yarım bırakmaz.

## 3. Test Durumu

Sandbox'ta gerçek Fireworks API'sine ağ erişimi yok (api.fireworks.ai allowlist
dışında) ve Docker binary'si yok — bu yüzden her şey ya (a) mock'lanmış
unit/entegrasyon testleriyle ya da (b) gerçek env var enjeksiyonu simülasyonuyla
(ağ çağrısı hariç) doğrulandı:

- [x] Config yükleme + rol otomatik ataması
- [x] Deterministic solver: pozitif + negatif (word problem'lere tahmin YOK) senaryolar
- [x] Triage heuristic: 8 kategorinin 7'si artık 0-token'la doğru sınıflandırılıyor
      (2 regex boşluğu bulunup düzeltildi: "discounted" ve "write a Python function")
- [x] Gemma-önce-dene-sonra-düş: başarı VE başarısızlık senaryoları, doğru token toplamı
- [x] Prompt compression: kod bloğu girinti/boşluğunun BİREBİR korunduğu doğrulandı
- [x] Validator: tüm kategori-özel kontroller (JSON, sayı, kod işareti, ret, kesilme)
- [x] Düzeltici retry: başarı VE başarısızlık (crash olmadan ilk cevaba geri dönme) yolları
- [x] main.py tam batch akışı: gerçek harness env var enjeksiyonu simüle edilerek
      (.env YOK, varsayılan /input→/output yolları) 8/8 görev işlendi, geçerli
      JSON yazıldı, exit code 0 — network tamamen kesikken bile

Test EDİLEMEDİ (kullanıcı kendi ortamında doğrulamalı):
- [ ] Docker image'ın gerçekten build olması (docker sandbox'ta yok)
- [ ] Gerçek Fireworks API çağrıları (network kısıtlaması)
- [ ] manage_gemma_deployment.py'nin gerçek API'ye karşı çalışması
- [ ] Gerçek model çıktılarının kalitesi (system prompt'lar hiç gerçek veriyle
      görülmedi — bu noktadan sonra kör uçuş riski var)

## 4. Kalan İşler / Öneriler

1. Kullanıcının kendi Fireworks API key'iyle yerel test koşusu — en yüksek öncelik.
2. README.md henüz yok — kurulum/çalıştırma talimatları eklenmeli.
3. Otomatik test paketi (pytest) henüz yok — şu ana kadarki testler hep ad-hoc
   `python3 -c "..."` komutlarıyla yapıldı, kalıcı tests/ klasörü yok.
4. Docker build/push süreci hiç denenmedi (docker buildx build --platform
   linux/amd64 ... komutu Dockerfile'da yorum olarak var ama çalıştırılmadı).
5. Gemma bonus'un tam $ tutarı ve tam koşulları teyit edilmedi.
6. Kategori-özel max_tokens değerleri (triage.py::CATEGORY_PROFILES) şu an makul
   tahminlerle sabitlendi — gerçek çıktılar görüldükten sonra ince ayar gerekebilir.

## 5. Devam Ederken Dikkat Edilmesi Gerekenler

- Kod stili: Tüm dosyalarda Türkçe (aksansız, ı/ş/ğ yerine i/s/g kullanılıyor)
  docstring/yorum kullanılıyor, İngilizce değişken/fonksiyon isimleri. Bu
  tutarlılığı koru.
- Her yeni dosya/değişiklik önce /home/claude/optiroute-ai/'de yapılıp test
  edilmeli, sonra /mnt/user-data/outputs/optiroute-ai/'a kopyalanıp present_files
  ile sunulmalı.
- Asla gerçek olmayan bir Fireworks API detayını (model ID, endpoint, fiyat) uydurma
  — hepsi bu oturumda web araması + docs.fireworks.ai sayfalarından doğrulandı.
- Kullanıcı "devam" dediğinde, en mantıklı sıradaki adımı SEÇİP uygulamaya geç,
  gereksiz yere tekrar onay sorma (zaten net bir yol haritası var).
