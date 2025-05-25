
import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
from detect_custom import run_detection

st.set_page_config(page_title="Deteksi Hilal", layout="centered")
st.title("🌙 Aplikasi Deteksi Hilal Otomatis")

st.markdown("Unggah **gambar atau video** untuk mendeteksi hilal menggunakan model YOLOv5.")

uploaded_file = st.file_uploader("Unggah berkas", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File terlalu besar (>10MB). Harap unggah file yang lebih kecil.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("⏳ Deteksi sedang diproses...")
    result_img_path, result_csv_path, df = run_detection(tmp_path)

    if df is not None and not df.empty:
        st.image(result_img_path, caption="🌓 Hasil Deteksi Hilal", use_column_width=True)
        st.success("✅ Hilal berhasil terdeteksi.")
        st.dataframe(df)
        st.download_button("📥 Unduh CSV", df.to_csv(index=False), file_name="hasil_deteksi.csv", mime="text/csv")
    else:
        st.warning("⚠️ Tidak ada hilal yang terdeteksi.")
