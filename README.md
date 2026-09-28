# Explainable Machine Learning for Breast Cancer Risk Prediction
### A Comparative Study of SVM, XGBoost, and Random Forest

Proyek ini membangun dan membandingkan tiga model klasifikasi machine learning — **Support Vector Machine (SVM)**, **XGBoost**, dan **Random Forest** — untuk memprediksi diagnosis kanker payudara (*Malignant*/Ganas vs *Benign*/Jinak) menggunakan **Wisconsin Breast Cancer Diagnostic Dataset**. Selain membandingkan performa, proyek ini menerapkan **Explainable AI (XAI)** menggunakan **SHAP** agar hasil prediksi model dapat diinterpretasikan secara transparan.

---

## 📁 Struktur Proyek

```
breast-cancer-project/
├── README.md                   # Dokumen ini
├── requirements.txt            # Daftar dependency Python
├── data/
│   └── breast-cancer.csv       # Dataset (569 sampel, 30 fitur numerik)
├── 01_EDA.ipynb                # Exploratory Data Analysis
├── 02_SVM.ipynb                # Model SVM (mandiri/independen)
├── 03_XGBoost.ipynb            # Model XGBoost (mandiri/independen)
├── 04_RandomForest.ipynb       # Model Random Forest (mandiri/independen)
├── 05_Model_Comparison.ipynb   # Perbandingan akhir ketiga model
├── results_svm.csv             # Output metrik SVM (dihasilkan oleh 02_SVM.ipynb)
├── results_xgboost.csv         # Output metrik XGBoost (dihasilkan oleh 03_XGBoost.ipynb)
└── results_rf.csv              # Output metrik Random Forest (dihasilkan oleh 04_RandomForest.ipynb)
```

**Mengapa dipisah per algoritma?**
Setiap notebook algoritma (`02`, `03`, `04`) bersifat **mandiri** — masing-masing memiliki blok *load data* dan *preprocessing* sendiri, sehingga bisa dijalankan/diubah/didebug sendiri-sendiri tanpa memengaruhi notebook lain. Ini memudahkan tracking jika ada isu spesifik pada satu algoritma, atau jika ingin menambahkan algoritma baru di kemudian hari.

---

## 📊 Tentang Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**
- 569 sampel pasien, 30 fitur numerik hasil digitalisasi citra *Fine Needle Aspirate* (FNA) dari massa payudara.
- Fitur mendeskripsikan karakteristik inti sel (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) — masing-masing dalam 3 varian: `_mean`, `_se` (standard error), dan `_worst`.
- Target: `diagnosis` → **M** (Malignant/Ganas) atau **B** (Benign/Jinak).
- Tidak ada missing value maupun data duplikat.

---

## ⚙️ Instalasi & Persiapan Environment

### 1. Buat virtual environment (opsional tapi disarankan)
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependency
```bash
pip install -r requirements.txt
```

### 3. Jalankan Jupyter Notebook
```bash
jupyter notebook
```
atau jika menggunakan JupyterLab:
```bash
jupyter lab
```

> Pastikan struktur folder tetap seperti di atas — semua notebook membaca dataset dengan path relatif `data/breast-cancer.csv`, jadi notebook harus dijalankan dari root folder `breast-cancer-project/`.

---

## ▶️ Urutan Menjalankan Notebook

| Urutan | Notebook | Keterangan |
|---|---|---|
| 1 | `01_EDA.ipynb` | Eksplorasi data: distribusi kelas, histogram, correlation heatmap, boxplot, pairplot |
| 2 | `02_SVM.ipynb` | Preprocessing → Tuning (GridSearchCV) → Evaluasi → SHAP → simpan `results_svm.csv` |
| 3 | `03_XGBoost.ipynb` | Preprocessing → Tuning → Evaluasi → SHAP + Feature Importance → simpan `results_xgboost.csv` |
| 4 | `04_RandomForest.ipynb` | Preprocessing → Tuning → Evaluasi → SHAP + Feature Importance → simpan `results_rf.csv` |
| 5 | `05_Model_Comparison.ipynb` | Menggabungkan ketiga `results_*.csv` menjadi tabel & grafik perbandingan akhir |

Notebook `02`, `03`, `04` **boleh dijalankan dalam urutan bebas** (tidak saling bergantung), namun **`05` wajib dijalankan paling akhir** karena membutuhkan file `results_*.csv` dari ketiganya.

---

## 🧠 Metodologi Singkat

1. **Preprocessing**: drop kolom `id`, encode label (`M`=1, `B`=0), split data 80/20 (stratified), standardisasi fitur (`StandardScaler`) khusus untuk SVM.
2. **Modeling**: setiap model di-tuning menggunakan `GridSearchCV` dengan 5-fold *Stratified Cross Validation*, dioptimasi terhadap skor **ROC-AUC**.
3. **Evaluasi**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, dan Kurva ROC pada data uji (test set) yang belum pernah dilihat model.
4. **Explainable AI**:
   - `TreeExplainer` (SHAP) untuk XGBoost & Random Forest — cepat dan eksak karena berbasis struktur pohon.
   - `KernelExplainer` (SHAP) untuk SVM — model-agnostic, karena SVM bukan model berbasis pohon.
   - Feature importance bawaan sebagai pembanding untuk model berbasis pohon.
   - SHAP waterfall plot untuk interpretasi prediksi pada level individu pasien.

---

## 🔧 Menambahkan Algoritma Baru

Jika ingin menambahkan algoritma ke-4 (misalnya Logistic Regression):
1. Buat notebook baru, misal `06_LogisticRegression.ipynb`, dengan struktur serupa `02`–`04` (load data → preprocessing → training → evaluasi → simpan `results_logreg.csv`).
2. Tambahkan baris berikut di `05_Model_Comparison.ipynb` pada bagian penggabungan hasil:
   ```python
   results_logreg = pd.read_csv('results_logreg.csv', index_col='Model')
   results_df = pd.concat([results_svm, results_xgb, results_rf, results_logreg])
   ```

---

## 📌 Catatan & Batasan

- Dataset relatif kecil (569 sampel) — disarankan validasi eksternal pada dataset lain untuk memperkuat generalisasi model.
- Terdapat multikolinearitas tinggi antar fitur turunan (`_mean`, `_se`, `_worst` dari radius/perimeter/area) — bisa dieksplorasi teknik *feature selection* lebih lanjut.
- Proyek ini bersifat edukatif/penelitian. Untuk penggunaan klinis nyata, diperlukan validasi lebih lanjut bersama tenaga medis serta uji coba prospektif — **bukan pengganti diagnosis medis profesional**.

---

## 📚 Referensi Dataset

Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository.