# 📱 SMS Spam Detection — Text Mining Project

Sebuah model *machine learning* yang membaca isi SMS dan menentukan apakah pesan itu **spam** atau **pesan sah (ham)**, lengkap dengan aplikasi web interaktif tempat siapa pun bisa menempel sebuah pesan dan langsung melihat hasilnya beserta alasannya.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit--learn](https://img.shields.io/badge/model-Logistic%20Regression-orange) ![Streamlit](https://img.shields.io/badge/app-Streamlit-red) ![Data Mining](https://img.shields.io/badge/domain-Text%20Mining-6c5ce7)

---

## 📄 Executive Summary

**Masalah.** Spam SMS bukan sekadar mengganggu — ia sering menjadi pintu masuk penipuan (*smishing*): tautan palsu, undian bohong, dan pencurian data. Menyaringnya secara manual tidak praktis karena volumenya besar dan polanya terus berubah.

**Solusi.** Proyek ini melatih **satu model machine learning** untuk mengklasifikasikan SMS secara otomatis. Model belajar dari 5.572 contoh pesan nyata, mengenali pola bahasa yang membedakan spam dari percakapan biasa, lalu memberikan prediksi disertai skor keyakinan.

**Cara kerja (ringkas).** Teks mentah diubah menjadi angka memakai teknik **TF-IDF** (memberi bobot pada kata yang khas), lalu sebuah **Logistic Regression** mempelajari kata mana yang menandai spam. Ini adalah alur *text mining* klasik — cabang sah dari data mining.

**Hasil.** Pada data uji yang belum pernah dilihat model:

| Metrik | Nilai | Arti praktis |
|---|---|---|
| **Spam F1-score** | **~0,92** | Keseimbangan baik antara menangkap spam & tidak salah tuduh |
| **Akurasi** | **~0,98** | 98 dari 100 pesan diklasifikasikan benar |
| **ROC-AUC** | **~0,99** | Kemampuan memisahkan spam vs ham hampir sempurna |
| **False positive** | **Sangat rendah** | Pesan penting nyaris tak pernah salah diblokir |

**Apa yang "ditambang".** Analisis model mengungkap pola yang jelas dan masuk akal:
- **Penanda spam:** kata transaksi & ajakan cepat — *txt, claim, won, prize, 150p, mobile, www, reply, stop*.
- **Penanda ham:** kosakata percakapan sehari-hari — *ok, home, hey, later, sorry*.

Inilah "kejadian data mining" yang sesungguhnya: model tidak dihafalkan aturan, melainkan **menemukan sendiri** ciri pembeda dari data.

**Deliverable.** (1) Notebook Colab berisi seluruh pipeline yang bisa dijalankan ulang, dan (2) aplikasi web **Streamlit** bernama *Smish*, tempat pengguna menempel SMS dan melihat verdict, skor spam, serta kata pemicu yang disorot — sehingga keputusan model **transparan**, bukan kotak hitam.

**Keterbatasan.** Model dilatih pada SMS berbahasa Inggris; performa pada bahasa lain akan menurun. Karena data timpang (~13% spam), evaluasi bertumpu pada F1, bukan sekadar akurasi. Prediksi bersifat probabilistik dan tetap bisa keliru pada pesan yang ambigu.

---

## 📂 Struktur Proyek

```
sms-spam-detection/
├── sms_spam_training.ipynb   # Notebook Colab: pipeline text mining lengkap
├── app.py                    # Aplikasi web Streamlit "Smish"
├── sms_spam_model.joblib     # Artefak model (dihasilkan notebook, ~108 KB)
├── requirements.txt          # Dependensi
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1 · Melatih model (Google Colab)

1. Buka `sms_spam_training.ipynb` di [Google Colab](https://colab.research.google.com/).
2. `Runtime ▸ Run all`. Dataset diunduh otomatis lewat `ucimlrepo` (ada *fallback* ke arsip resmi UCI bila API bermasalah).
3. Di akhir, berkas **`sms_spam_model.joblib`** terunduh otomatis.

### 2 · Menjalankan aplikasi web (lokal)

```bash
pip install -r requirements.txt
# letakkan sms_spam_model.joblib di folder yang sama dengan app.py
streamlit run app.py
```

Buka `http://localhost:8501`, tempel sebuah SMS, tekan **Periksa pesan**. Tersedia tombol contoh (spam / biasa / smishing) untuk mencoba cepat.

---

## 🔬 Metodologi (Alur Text Mining)

1. **Data Understanding** — memeriksa jumlah pesan, proporsi ham/spam, duplikat, dan panjang pesan.
2. **EDA** — distribusi kelas, distribusi panjang pesan per kelas, dan *word cloud* untuk melihat kosakata dominan.
3. **Data Preparation** — **TF-IDF** (unigram + bigram, buang *stopwords*, `min_df=2`) mengubah teks menjadi vektor; *stratified* split 80/20; vectorizer dipasang dalam `Pipeline` untuk mencegah *data leakage*.
4. **Modeling** — membandingkan dua baseline teks (Naive Bayes vs Logistic Regression), lalu memilih **Logistic Regression** (`class_weight='balanced'`) sebagai model final.
5. **Evaluation** — precision/recall/F1 pada kelas spam, confusion matrix, kurva ROC, dan validasi silang 5-fold.
6. **Interpretation** — membaca koefisien model untuk mengungkap kata pemicu spam vs ham.

### Catatan metodologis
- **Ketidakseimbangan kelas** (~13% spam) ditangani dengan `class_weight='balanced'` dan dinilai memakai **F1/ROC-AUC**, bukan akurasi mentah yang menyesatkan pada data timpang.
- **Kenapa Logistic Regression?** F1 stabil, precision/recall seimbang, dan koefisiennya langsung dapat diinterpretasi — cocok untuk menjelaskan *mengapa* sebuah pesan ditandai spam (fitur transparansi di aplikasi).

---

## 📊 Ringkasan Hasil

| Aspek | Detail |
|---|---|
| Dataset | SMS Spam Collection — UCI (ID 228), 5.572 pesan, 13,4% spam, 0 *missing* |
| Representasi | TF-IDF (unigram + bigram) |
| Model | Logistic Regression (`class_weight='balanced'`) |
| Spam F1 (test) | ~0,92 |
| Akurasi (test) | ~0,98 |
| ROC-AUC (test) | ~0,99 |

---

## 📚 Referensi

- Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG).
- SMS Spam Collection — UCI Machine Learning Repository (ID 228): <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>

## 📝 Lisensi
Dataset dirilis dengan lisensi **CC BY 4.0**. Kode proyek bebas dipakai untuk keperluan akademik dengan atribusi.
