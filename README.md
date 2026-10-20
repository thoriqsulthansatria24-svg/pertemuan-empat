# Studi Kasus: Pipeline Machine Learning Cuaca Regional Sumatera–Kedah

Berdasarkan kuliah umum **"Introduction to Machine Learning"** dalam rangkaian *Informatics Professional Lecture Series 2026* oleh **Dr. Taniza Tajudin** (Senior Lecturer, Faculty of Computer and Mathematical Sciences, Universiti Teknologi MARA / UiTM Cawangan Kedah, Malaysia — Kolaborasi Internasional **Universitas Bengkulu & UiTM**).

Pesan utama materi: berbeda dari pemrograman tradisional yang mengandalkan aturan eksplisit dari programmer, *Machine Learning* mempelajari pola langsung dari data melalui pendekatan *Supervised Learning*, *Unsupervised Learning*, *Semi-Supervised*, dan *Reinforcement Learning*, dengan tetap mengedepankan evaluasi akurasi serta etika AI (mitigasi bias, privasi data, dan transparansi).

## Hubungan dengan Materi

| Poin Materi Kuliah Umum | Implementasi pada Proyek Ini |
| :--- | :--- |
| **Pemrograman Tradisional vs *Machine Learning*** | Eksperimen komparatif antara aturan statis (`if-else` kelembapan) dan model ML pada data cuaca nyata, lengkap dengan metrik selisih akurasi |
| **Supervised Learning (*Regression* & *Classification*)** | *Regression* memprediksi suhu udara numerik (°C) dengan evaluasi MAE & $R^2$; *Classification* mengelompokkan kategori *Potensi Hujan* vs *Cerah/Berawan* |
| **Unsupervised Learning (*Clustering*, *PCA*, *Anomaly Detection*)** | Algoritma **K-Means** mengelompokkan 3 rezim mikroklimat tanpa label, **PCA** mereduksi dimensi fitur ke 2 komponen utama, dan Z-Score multivariat mendeteksi **Anomali Cuaca** |
| **Etika ML (Bias, Privasi & Transparansi)** | Menampilkan bobot kontribusi fitur (*Explainable AI*) dan mengaudit ketimpangan error prediksi antara stasiun Sumatera (UNIB Bengkulu dkk.) dan Kedah (UiTM Malaysia) |
| **Pola Pikir *Data-Driven* & Otomasi** | GitHub Actions menarik data terbaru dari Open-Meteo API, melatih ulang model, dan memperbarui dashboard web setiap 6 jam |

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
python pipeline_ml_cuaca.py
