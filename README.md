# IF3170_G13_brownbear

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027 - **Supervised Learning**.

Implementasi *from scratch* algoritma KNN, Decision Tree Learning, Logistic Regression (L1/L2/Elastic Net), SVM (linear/polynomial/gaussian), dan Naive Bayes untuk memprediksi pembatalan pemesanan penginapan, serta perbandingannya dengan scikit-learn.

## Struktur Repository

```
.
├── data/                 # dataset Kaggle
├── docs/                 # laporan (.pdf)
├── output/               # hasil run algoritma (model tersimpan, submission, gambar)
├── src/
│   ├── Notebook_Template_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
│   └── brownbear/
│       ├── preprocessing.py
│       └── models/       # KNN, DTL, LogReg, SVM, Naive Bayes (from scratch)
├── pyproject.toml
└── uv.lock
```

## Setup

### Prasyarat

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) terlebih dahulu. Cek apakah sudah terpasang dengan `uv --version`. Jika belum, jalankan perintah sesuai sistem operasi:

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS dan Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Buka ulang terminal setelah instalasi agar perintah `uv` dikenali.

### Instalasi Project

```bash
git clone https://github.com/<username>/IF3170_G13_brownbear.git
cd IF3170_G13_brownbear
uv sync
```

`uv sync` akan memasang Python 3.13 (jika belum ada), membuat `.venv`, dan memasang semua dependency sesuai `uv.lock`.

## Cara Menjalankan

```bash
# Jupyter
uv run jupyter notebook src/IF3170_G13_brownbear
```

Atau buka `src/IF3170_G13_brownbear.ipynb` di VS Code dan pilih kernel `.venv`.

Menambah dependency baru: `uv add <nama-paket>` lalu commit `pyproject.toml` dan `uv.lock`.

## Pembagian Tugas

| NIM | Nama | Tugas |
|-----|------|-------|
| 13524016    | Wafiq Hibban Robbany     |       |
| 13524025    | Moh. Hafizh Irham Perdana     |       |
| 13524054    | Wildan Abdurrahman Ghazali     |       |
| 13524070    | A. Fawwaz Azam Wicaksono     |       |
| 13524124    | Zahran Alvan Putra Winarko     |       |
