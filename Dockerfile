# OptiRoute AI - Track 1: General-Purpose AI Agent
#
# ONEMLI (build zamaninda): Judging VM linux/amd64 calisiyor. Apple Silicon
# (M1/M2/M3) uzerinde build ediyorsan MUTLAKA su komutu kullan, yoksa image
# pull edilemez ve sifir puan alirsin:
#
#   docker buildx build --platform linux/amd64 --tag <senin-imajin>:latest --push .
#
# Intel/AMD makinede veya GitHub Actions'ta standart 'docker build' zaten
# linux/amd64 uretir, ekstra bir sey gerekmez.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Once sadece requirements'i kopyala -> Docker layer cache'i sayesinde
# kod degistiginde bagimliliklar yeniden kurulmaz (build hizi icin onemli).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
