#!/bin/bash

# Pastikan script dijalankan dari root project directory
cd "$(dirname "$0")/.."

OUTPUT_FILE="client_analysis_results.tar.gz"

echo "[INFO] Mengumpulkan file gambar, laporan, dan feature numpy..."

# Tarball direktori dan file spesifik
# 2> /dev/null digunakan agar jika ada file yang belum ter-generate, tar tidak gagal total
tar -czvf $OUTPUT_FILE \
    reports/figures/limuc_feature_importance.png \
    reports/figures/limuc_umap_comparison.png \
    reports/figures/tmc_feature_importance.png \
    reports/figures/tmc_umap_comparison.png \
    reports/limuc_thresholds.csv \
    reports/tmc_thresholds.csv \
    data/limuc_features/ \
    data/tmc_features/ 2>/dev/null

if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo "✅ Berhasil! File arsip siap didownload:"
    echo "📦 $(pwd)/$OUTPUT_FILE"
    echo "=========================================================="
    echo "Gunakan perintah 'scp' di terminal lokal laptop Anda untuk mendownloadnya:"
    echo "scp D13K48009@DGXH100:$(pwd)/$OUTPUT_FILE ./"
else
    echo "❌ Gagal membuat arsip. Pastikan direktori reports/ dan data/ ada."
fi
