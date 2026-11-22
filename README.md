# IF3170_G13_brownbear

Repo ini berisi pengerjaan Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027 oleh kelompok G13 Brownbear. Kasus yang kami kerjakan adalah memprediksi apakah sebuah pemesanan penginapan akan dibatalkan oleh tamunya (`is_canceled`). Dengan prediksi ini, pihak penginapan bisa memperkirakan berapa kamar yang benar-benar akan terisi. Kamar yang kemungkinan besar dibatalkan bisa dijual lagi, tetapi penginapan juga tidak boleh sampai *overbooking*.

Dataset diambil dari kompetisi Kaggle kelas. Isinya campuran fitur numerik dan kategorikal, misalnya jumlah tamu, lama menginap, harga per malam, jarak hari dari pemesanan ke kedatangan, paket makan, tipe kamar, dan kanal pemesanan. Beberapa kolom juga punya nilai kosong. Karena itu, setiap model kami beri pipeline persiapan data sendiri sesuai kebutuhannya. Semua statistik preprocessing hanya dihitung dari data latih.

Lima algoritma *supervised learning* kami tulis sendiri dari nol di package `brownbear`, dengan NumPy hanya untuk perhitungan. Setelah itu, setiap model dibandingkan dengan versi scikit-learn-nya di notebook.

## Model

**KNN.** Mendukung jarak Euclidean, Manhattan, dan Minkowski, dengan voting seragam atau berbobot jarak. Untuk dataset ini kami menambahkan penalti jarak jika atribut diskret dua pemesanan berbeda, dan imputasi `days_before_arrival` dari pemesanan lain dengan tanggal dan lama menginap yang sama. Bobot setiap fitur dicari dengan *coordinate descent* pada 5-fold cross-validation.

**Decision Tree Learning.** Kami memilih CART dengan kriteria Gini. Pohonnya bisa langsung menerima atribut numerik, atribut kategorikal (split biner pada himpunan kategori), dan nilai kosong. Supaya tidak *overfit*, pohon dipangkas dengan *minimal cost-complexity pruning*. Nilai `ccp_alpha` dipilih lewat cross-validation.

**Logistic Regression.** Mendukung regularisasi L1, L2, dan Elastic Net. Karena L1 tidak bisa diturunkan di titik nol, pelatihannya memakai *proximal gradient* yang dipercepat dengan FISTA dan *adaptive restart*, sehingga konvergensinya jauh lebih cepat daripada *gradient descent* biasa.

**SVM.** Mendukung kernel linear, polynomial, dan Gaussian (RBF). Masalah dual-nya diselesaikan dengan SMO. Pasangan multiplier yang dioptimasi di setiap iterasi dipilih dengan *working set selection* orde dua (Fan, Chen, dan Lin, 2005).

**Naive Bayes.** Naive Bayes kategorikal dengan *additive smoothing*. Fitur kontinu didiskretisasi terlebih dahulu. Kami juga membuat Tree-Augmented Naive Bayes (TAN), yang mengizinkan setiap fitur bergantung pada satu fitur lain. Struktur pohon dependensinya dibangun dari *conditional mutual information* antarfitur.

Semua model mewarisi kelas `BaseModel`, sehingga bisa disimpan dan dimuat kembali dengan `model.save(path)` dan `Model.load(path)`.

## Hasil

Berikut macro F1 tiap model pada data validasi (20% dari `train.csv`):

| Model | Macro F1 |
|-------|----------|
| KNN | 0.862 |
| Decision Tree (CART) | 0.839 |
| SVM Gaussian | 0.822 |
| SVM Polynomial | 0.794 |
| Logistic Regression L1 | 0.790 |
| Naive Bayes TAN | 0.778 |
| SVM Linear | 0.768 |
| Naive Bayes | 0.749 |

KNN memberi hasil terbaik, jadi ketiga submisi Kaggle kami (`submission1.csv` sampai `submission3.csv`) memakai KNN dengan konfigurasi yang sedikit berbeda. Prediksi model *from scratch* hampir selalu sama dengan versi scikit-learn-nya. Tingkat kesamaannya 96% untuk KNN dan di atas 98% untuk model lainnya.

Selain itu, notebook juga membahas beberapa hal berikut:

- gambar percabangan pohon dengan parameter top-N
- video kontur log-loss beserta lintasan parameter (θ₀, θ₁) selama training
- gambar bidang pemisah, margin, dan support vector untuk ketiga kernel SVM
- graf Bayesian Network dari TAN
- analisis *cost-sensitive classification* untuk kasus *false positive* yang dua kali lebih mahal daripada *false negative*

## Struktur repo

```text
.
├── data/                               # dataset kompetisi (train, test, contoh submission)
├── docs/                               # laporan
├── output/                             # hasil run notebook
├── src/
│   ├── IF3170_G13_brownbear.ipynb      # notebook utama
│   └── brownbear/                      # package berisi model dan utilitas
│       ├── preprocessing.py            # pipeline persiapan data tiap model
│       ├── models/
│       │   ├── base.py                 # kelas dasar fit/predict dan save/load
│       │   ├── knn.py
│       │   ├── dtl.py                  # CART dan cost-complexity pruning
│       │   ├── logistic_regression.py  # L1, L2, Elastic Net dengan FISTA
│       │   ├── svm.py                  # kernel linear, polynomial, Gaussian dengan SMO
│       │   └── naive_bayes.py          # Naive Bayes dan TAN
│       └── visualization/
│           ├── decision_tree.py        # gambar pohon top-N
│           ├── logistic_regression.py  # video kontur log-loss
│           ├── svm.py                  # bidang pemisah, margin, support vector
│           └── bayesian_network.py     # graf Bayesian Network
├── pyproject.toml
└── uv.lock
```

## Setup

Project ini memakai Python 3.13 dan [uv](https://docs.astral.sh/uv/) untuk mengelola dependensi. Semua library yang dibutuhkan tercantum di `pyproject.toml`, dan versinya dikunci di `uv.lock`.

```bash
git clone https://github.com/Alvacodee/IF3170_G13_brownbear.git
cd IF3170_G13_brownbear
uv sync
```

Dataset kompetisi sudah ada di folder `data/`, jadi tidak perlu diunduh lagi.

## Cara menjalankan

Buka notebook lalu jalankan semua sel dari atas sampai bawah (*Run All*):

```bash
uv run jupyter notebook src/IF3170_G13_brownbear.ipynb
```

Notebook juga bisa dijalankan langsung dari terminal:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace src/IF3170_G13_brownbear.ipynb
```

Satu kali run penuh memakan waktu sekitar 40 menit. Bagian yang paling lama adalah cross-validation untuk tuning dan training SVM. Seed sudah diatur di awal notebook, jadi setiap run menghasilkan angka dan file submisi yang sama.

Setiap kali notebook dijalankan, hasilnya ditulis ke folder `output/`. Isinya model yang sudah dilatih (`.pkl`), gambar dan video visualisasi, serta tiga file submisi Kaggle. Hasil run terakhir kami juga sudah tersimpan di folder ini, jadi model bisa langsung dimuat tanpa training ulang. Contohnya, dari dalam folder `src/`:

```python
from brownbear.models import KNN

model = KNN.load("../output/knn.pkl")
```

Pencarian bobot fitur KNN butuh waktu lama, jadi kami matikan secara default (`RUN_WEIGHT_SEARCH = False`). Bobot hasil pencarian terakhir sudah disimpan di `KNN_FEATURE_WEIGHTS` pada `preprocessing.py`. Ubah nilainya menjadi `True` jika ingin mengulang pencarian dari awal.

## Pembagian tugas

| NIM | Nama | Tugas |
|-----|------|-------|
| 13524016 | Wafiq Hibban Robbany | KNN |
| 13524025 | Moh. Hafizh Irham Perdana | Decision Tree Learning (CART), pruning, dan gambar pohon top-N |
| 13524054 | Wildan Abdurrahman Ghazali | Logistic Regression dan video kontur log-loss |
| 13524070 | A. Fawwaz Azam Wicaksono | Naive Bayes dan TAN, pipeline preprocessing, tuning KNN, integrasi notebook, dan submisi Kaggle |
| 13524124 | Zahran Alvan Putra Winarko | SVM, gambar kernel SVM, dan setup repo |

## Referensi

- [CART dan Gini](https://scikit-learn.org/stable/modules/tree.html)
- [KNN berbobot jarak](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dan [pemrosesan jarak per batch](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html)
- [Optimasi proximal](https://web.stanford.edu/~boyd/papers/prox_algs.html), [akselerasi FISTA](https://epubs.siam.org/doi/10.1137/080716542), dan [restart adaptif](https://doi.org/10.1007/s10208-013-9150-3) untuk Logistic Regression
- [Pemilihan working set orde dua](https://jmlr.org/papers/v6/fan05a.html) untuk SMO pada SVM, dibandingkan dengan [pasangan pelanggar maksimum](https://doi.org/10.1162/089976601300014493)
- [Naive Bayes kategorikal](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes)
- [Tree-Augmented Naive Bayes](https://doi.org/10.1023/A:1007465528199), [pohon dependensi Chow-Liu](https://doi.org/10.1109/TIT.1968.1054142), dan [algoritma Prim](https://doi.org/10.1002/j.1538-7305.1957.tb01515.x) untuk struktur Naive Bayes
- [Preprocessing dan split](https://scikit-learn.org/stable/modules/compose.html)
- [Jarak campuran numerik dan kategorikal](https://doi.org/10.2307/2528823) untuk penalti jarak KNN
- [Imputasi berbasis tetangga terdekat](https://doi.org/10.1080/713827181) untuk `days_before_arrival`
- [Coordinate descent](https://doi.org/10.1007/s10107-015-0892-3) untuk pencarian bobot fitur KNN
