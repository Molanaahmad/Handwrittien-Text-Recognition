# Handwritten Text Recognition untuk Digitalisasi Catatan Transaksi Manual

## 📌 Deskripsi

Repository ini berisi implementasi penelitian skripsi mengenai **simulasi otomatisasi digitalisasi catatan transaksi manual menggunakan Handwritten Text Recognition (HTR) berbasis Deep Learning**.

Penelitian ini memanfaatkan model **TrOCR (Transformer-based Optical Character Recognition)** untuk mengenali teks tulisan tangan dari catatan transaksi manual dan mengubahnya menjadi data teks yang dapat diproses secara digital.

Sistem dikembangkan sebagai simulasi proses digitalisasi catatan transaksi pada lingkungan koperasi/kantin kampus, mulai dari input gambar tulisan tangan, preprocessing, pengenalan teks menggunakan model TrOCR, evaluasi hasil prediksi, hingga penyajian hasil pada aplikasi.

## 🎯 Tujuan

Penelitian ini bertujuan untuk:

* Menerapkan teknologi Handwritten Text Recognition (HTR) untuk mengenali catatan transaksi tulisan tangan.
* Mengimplementasikan model TrOCR berbasis Transformer untuk pengenalan teks tulisan tangan.
* Melakukan preprocessing dan augmentasi data untuk membantu proses pengenalan tulisan.
* Mengevaluasi performa model menggunakan Character Error Rate (CER), Word Error Rate (WER), dan Character Accuracy.
* Mensimulasikan proses digitalisasi catatan transaksi manual menjadi data digital.

## 🔄 Alur Sistem

```text
Catatan Transaksi Tulisan Tangan
              │
              ▼
       Input Gambar
              │
              ▼
        Preprocessing
   ┌─────────────────────┐
   │ Grayscale            │
   │ Thresholding         │
   │ Resize                │
   │ Noise Reduction       │
   └─────────────────────┘
              │
              ▼
      Data Augmentation
              │
              ▼
        Dataset Split
              │
              ▼
      Fine-Tuning TrOCR
              │
              ▼
       Text Recognition
              │
              ▼
       Hasil Prediksi OCR
              │
              ▼
      Evaluasi Model
   ┌─────────────────────┐
   │ CER                  │
   │ WER                  │
   │ Character Accuracy   │
   └─────────────────────┘
              │
              ▼
    Simulasi Digitalisasi
     Catatan Transaksi
```

## 🧠 Model

Model utama yang digunakan dalam penelitian ini adalah:

**TrOCR (Transformer-based Optical Character Recognition)**

TrOCR merupakan pendekatan OCR berbasis Transformer yang menggabungkan komponen vision encoder dan text decoder untuk melakukan pengenalan teks dari gambar.

Model yang digunakan dalam penelitian:

```text
microsoft/trocr-base-handwritten
```

Model tersebut digunakan sebagai model dasar dan disesuaikan dengan dataset tulisan tangan yang digunakan dalam penelitian.

## 📊 Dataset

Dataset yang digunakan berupa gambar **catatan transaksi tulisan tangan** yang dikumpulkan untuk kebutuhan penelitian.

Setiap gambar memiliki label teks sebagai ground truth yang digunakan dalam proses training dan evaluasi.

Struktur dataset secara umum:

```text
dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   ├── image_003.png
│   └── ...
│
└── labels.txt
```

> Dataset penelitian tidak seluruhnya disertakan dalam repository apabila terdapat pertimbangan privasi atau keterbatasan distribusi data.

## 🛠️ Teknologi yang Digunakan

### Programming Language

* Python

### Deep Learning & OCR

* PyTorch
* Hugging Face Transformers
* TrOCR
* Hugging Face Datasets

### Image Processing

* OpenCV
* Pillow
* NumPy

### Evaluation

* Character Error Rate (CER)
* Word Error Rate (WER)
* Character Accuracy

### Application

* PyQt5
* SQLite

### Development Environment

* Google Colab
* Google Drive
* Visual Studio Code

## ⚙️ Preprocessing

Tahapan preprocessing digunakan untuk mempersiapkan citra tulisan tangan sebelum diproses oleh model TrOCR.

Tahapan yang digunakan meliputi:

1. Grayscale
2. Thresholding
3. Noise reduction
4. Resize
5. Image augmentation

Tahapan preprocessing dilakukan untuk menghasilkan citra yang lebih sesuai dengan kebutuhan model dan meningkatkan konsistensi data masukan.

## 🔬 Training & Evaluation

Model dilatih menggunakan dataset tulisan tangan yang telah melalui tahapan preprocessing.

Evaluasi dilakukan menggunakan **5-Fold Cross Validation** untuk memperoleh gambaran performa model pada beberapa pembagian dataset.

Metrik evaluasi yang digunakan:

### Character Error Rate (CER)

CER digunakan untuk mengukur tingkat kesalahan prediksi pada tingkat karakter.

Semakin rendah nilai CER, semakin baik kemampuan model dalam mengenali karakter.

### Word Error Rate (WER)

WER digunakan untuk mengukur kesalahan pengenalan pada tingkat kata.

Semakin rendah nilai WER, semakin baik hasil pengenalan teks.

### Character Accuracy

Character Accuracy digunakan untuk menggambarkan tingkat karakter yang berhasil dikenali dengan benar.

Semakin tinggi nilai Character Accuracy, semakin baik performa model.

## 🖥️ Simulasi Aplikasi

Hasil penelitian diimplementasikan dalam bentuk aplikasi desktop menggunakan **PyQt5**.

Fitur utama aplikasi meliputi:

* Upload gambar catatan transaksi
* Preprocessing gambar
* OCR menggunakan model TrOCR
* Menampilkan hasil pengenalan teks
* Parsing hasil OCR menjadi data transaksi
* Menampilkan confidence
* Menyimpan hasil ke database
* Melihat data transaksi
* Export hasil transaksi

Contoh alur penggunaan:

```text
Upload Image
     ↓
Preprocessing
     ↓
TrOCR Prediction
     ↓
OCR Result
     ↓
Text Parsing
     ↓
Transaction Data
     ↓
SQLite Database
```

## 📁 Struktur Repository

```text
## 📁 Struktur Repository

```text
skripsi-htr-trocr/
│
├── best_kfold_model/
│   └── Model TrOCR hasil eksperimen terbaik
│
├── Data test/
│   └── Data yang digunakan untuk pengujian sistem
│
├── main2.py
│   └── Program utama aplikasi simulasi HTR
│
├── requirements.txt
│   └── Daftar library yang dibutuhkan
│
├── README.md
│   └── Dokumentasi repository
│
└── .gitignore
    └── Daftar file/folder yang tidak diunggah ke repository
```

> Struktur folder dapat disesuaikan dengan struktur kode penelitian yang digunakan.

## 🚀 Instalasi

Clone repository:

```bash
git clone https://github.com/USERNAME/skripsi-htr-trocr.git
```

Masuk ke folder repository:

```bash
cd skripsi-htr-trocr
```

Install dependency:

```bash
pip install -r requirements.txt
```

## ▶️ Menjalankan Aplikasi

Setelah seluruh dependency terinstall, aplikasi dapat dijalankan dengan:

```bash
python application/app.py
```

Pastikan model dan dependency yang diperlukan telah tersedia sebelum menjalankan aplikasi.

## 📈 Hasil Penelitian

Eksperimen dilakukan menggunakan **5-Fold Cross Validation** untuk mengevaluasi konsistensi performa model.

Metrik yang digunakan:

| Metrik             | Keterangan                                              |
| ------------------ | ------------------------------------------------------- |
| CER                | Mengukur kesalahan pada tingkat karakter                |
| WER                | Mengukur kesalahan pada tingkat kata                    |
| Character Accuracy | Mengukur persentase karakter yang dikenali dengan benar |


## 🎓 Konteks Penelitian

Proyek ini dikembangkan sebagai bagian dari penelitian skripsi:

**"Simulasi Otomatisasi Digitalisasi Catatan Transaksi Manual Menggunakan Handwritten Text Recognition (HTR) Berbasis Deep Learning."**

Penelitian berfokus pada penerapan teknologi **Deep Learning dan Handwritten Text Recognition** untuk membantu proses digitalisasi data yang sebelumnya dicatat secara manual menggunakan tulisan tangan.

## 👨‍💻 Author

**Ahmad Maulana**

S1 Teknik Informatika
Universitas Nusantara PGRI Kediri

---

## 📄 License

Repository ini dibuat untuk kebutuhan **penelitian, akademik, pembelajaran, dan pengembangan portofolio**.

Silakan mencantumkan sumber/referensi apabila menggunakan bagian dari penelitian atau kode dalam repository ini.
