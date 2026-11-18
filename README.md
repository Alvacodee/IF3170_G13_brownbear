# IF3170_G13_brownbear

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027: klasifikasi pembatalan pemesanan penginapan. KNN, Decision Tree Learning, Logistic Regression, SVM, dan Naive Bayes diimplementasikan dari awal dan dibandingkan dengan scikit-learn.

## Struktur repo

```text
.
├── data/                         # train, test, dan contoh submission
├── docs/                         # laporan
├── output/                       # model, gambar, dan submission hasil run
├── src/
│   ├── IF3170_G13_brownbear.ipynb
│   └── brownbear/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── visualization/        # gambar dan video bonus
│       │   ├── __init__.py
│       │   ├── _style.py         # warna dan helper gambar bersama
│       │   ├── bayesian_network.py
│       │   ├── decision_tree.py  # top-N split pohon keputusan
│       │   ├── logistic_regression.py  # video kontur log-loss
│       │   └── svm.py            # bidang pemisah, margin, dan support vector
│       └── models/
│           ├── __init__.py
│           ├── base.py           # fit/predict dan save/load bersama
│           ├── knn.py
│           ├── dtl.py
│           ├── logistic_regression.py
│           ├── svm.py
│           └── naive_bayes.py    # Naive Bayes dan Tree-Augmented Naive Bayes
├── pyproject.toml
└── uv.lock
```

## Persiapan

```bash
uv sync
uv run jupyter notebook
```

Dependensi dikelola melalui `pyproject.toml` dan dikunci di `uv.lock`. Preprocessing di-fit pada data latih, lalu dipakai untuk mentransformasi data validasi dan data uji.

## Referensi implementasi eksternal

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

## Anggota

| NIM | Nama |
|-----|------|
| 13524016 | Wafiq Hibban Robbany |
| 13524025 | Moh. Hafizh Irham Perdana |
| 13524054 | Wildan Abdurrahman Ghazali |
| 13524070 | A. Fawwaz Azam Wicaksono |
| 13524124 | Zahran Alvan Putra Winarko |
