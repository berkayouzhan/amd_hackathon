# Adaptive Model Dispatcher - Track 1: General-Purpose AI Agent
#
# ONEMLI (build zamaninda): Judging VM linux/amd64 calisiyor. Apple Silicon
# (M1/M2/M3) uzerinde build ediyorsan MUTLAKA su komutu kullan, yoksa image
# pull edilemez ve sifir puan alirsin:
#
#   docker buildx build --platform linux/amd64 --tag <senin-imajin>:latest --push .
#
# Intel/AMD makinede veya GitHub Actions'ta standart 'docker build' zaten
# linux/amd64 uretir, ekstra bir sey gerekmez.

# === STAGE 1: Builder ===
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Yerel model dosyasini HuggingFace uzerinden indir (1.1GB)
# Ayri bir katman olarak: sadece model URL degisirse tekrar indirilir
RUN curl -L -o /build/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Python bagimliliklerini kur (build-essential ile derleme gerekebilir)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === STAGE 2: Runtime ===
FROM python:3.11-slim

WORKDIR /app

# Builder'dan kurulan Python paketlerini kopyala
COPY --from=builder /install /usr/local

# Builder'dan indirilen model dosyasini kopyala
COPY --from=builder /build/qwen2.5-1.5b-instruct-q4_k_m.gguf /app/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Uygulama dosyalarini kopyala
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Harness /input/tasks.json'i buraya mount edecek, /output/results.json'i
# buraya yazmamizi bekleyecek (Participant Guide). Dizinleri onceden
# olusturuyoruz; harness bunlarin uzerine bind-mount yapabilir.
#
# NOT (bilerek root olarak calisiyoruz): Harness'in /input ve /output'u
# hangi UID/izinle mount edecegini bilmiyoruz. Non-root bir kullaniciya
# gecmek, host tarafinin mount izinleriyle uyusmazsa "Permission denied"
# ile results.json yazamama riski yaratir - ki bu durum "malformed/eksik
# output = sifir puan" kuralina carpar. Bu kisa omurlu, tek amacli
# yarisma container'i icin, guvenlik best-practice'inden (non-root) daha
# once gelen oncelik: HER KOSULDA cikti yazabilmek.
RUN mkdir -p /input /output

ENTRYPOINT ["python", "main.py"]

