# Panduan Eksekusi Analisis Presentasi Klien di Server DGX

Panduan ini berisi cara mengeksekusi script analisis tekstur (*Texture Analysis*) untuk keperluan presentasi klien di dalam *environment* server DGX (D13K48009).

## 1. Lokasi File
Script utama telah disesuaikan letaknya menjadi:
`scripts/client_presentation_analysis.py`

## 2. Persyaratan Data
Pastikan bahwa dataset telah tersedia pada *path* bawaan server DGX kita:
- **LIMUC**: `/raid/D13K48009/texture/LIMUC`
- **TMC**: `/raid/D13K48009/texture/TMC`

*(Script akan membaca file secara otomatis dari directory di atas, dan untuk TMC membaca split `train.txt` dan `test.txt` yang sudah disiapkan).*

## 3. Cara Menjalankan
Buka terminal pada server DGX Anda, masuk ke dalam folder *root* proyek (Colonomind-Texture-analysis), lalu jalankan perintah berikut:

```bash
python scripts/client_presentation_analysis.py
```

## 4. Output yang Dihasilkan
Script ini menggunakan mode *Headless* (`matplotlib.use('Agg')`), sehingga tidak akan menampilkan GUI. Semua hasil analisis akan tersimpan otomatis:

1. **Features Extracted (.npy)**
   Tersimpan di:
   - `data/limuc_features/`
   - `data/tmc_features/`
   
2. **Visualisasi Gambar (.png)**
   Tersimpan di `reports/figures/` yang berisi:
   - Feature Importance Bar Charts
   - UMAP Comparison Scatter Plots
   - Cluster Scatter Plots
   - Box Plots for Edge Sharpness
   
3. **Thresholds Rule-Based (.csv)**
   Tersimpan di `reports/`:
   - `limuc_thresholds.csv`
   - `tmc_thresholds.csv`

Anda bisa menggunakan perintah `scp` (Secure Copy) atau rsync untuk mengunduh folder `reports/figures/` kembali ke komputer lokal Anda setelah eksekusi selesai.
