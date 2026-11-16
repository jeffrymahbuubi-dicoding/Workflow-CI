# Workflow-CI

Pipeline CI untuk model prediksi kegagalan mesin (dataset AI4I 2020, lisensi CC BY 4.0).

## Isi

- `MLProject/` : proyek MLflow (`MLproject`, `conda.yaml`, `modelling.py`, data hasil prapemrosesan).
- `.github/workflows/ci.yml` : workflow GitHub Actions.
- `mlruns/` : hasil pelatihan, di-commit oleh workflow.
- `MLProject/DockerHub.txt` : tautan image di Docker Hub.

## Alur workflow

1. Checkout kode dan siapkan Python 3.12.7.
2. Pasang dependensi (versi sama dengan `conda.yaml`).
3. `mlflow run MLProject --env-manager=local` melatih model.
4. Ambil `run_id` terbaru dari `mlruns/0/`.
5. Unggah `mlruns/` sebagai artifact GitHub dan commit ke repositori.
6. `mlflow models build-docker`, lalu tag dan push ke Docker Hub (`latest` dan SHA commit).

Secrets yang dibutuhkan: `DOCKER_HUB_USERNAME`, `DOCKER_HUB_ACCESS_TOKEN`.

## Menjalankan di komputer sendiri

```
pip install mlflow==2.19.0 scikit-learn==1.6.0 pandas==2.2.3 numpy==2.2.1 scipy==1.14.1 cloudpickle==3.1.0
mlflow run MLProject --env-manager=local
```

Parameter: `-P n_estimators=200 -P max_depth=12 -P dataset=ai4i2020_preprocessing`.
