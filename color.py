import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dominant Color Picker", layout="centered")

st.markdown("""
    <style>
    .color-box {
        width: 100%%;
        height: 100px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .hex-label {
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin-top: -10px;
        color: #333;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 14px;
        margin-top: 50px;
    }
    .intro {
        text-align: center;
        color: #444;
        font-size: 18px;
        margin-top: 20px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎨 Dominant Color Picker")
st.caption("Dapatkan 5 warna dominan dari gambar favoritmu secara instan!")

uploaded_file = st.file_uploader("📤 Upload Gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

def get_dominant_colors(image, k=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    return colors

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

st.markdown("---")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📷 Gambar yang diunggah", use_container_width=True)

    with st.spinner("🔍 Mengambil warna dominan..."):
        colors = get_dominant_colors(image, k=5)

    st.markdown("---")
    st.subheader("🌈 Palet Warna Dominan")

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        color_rgb = colors[idx]
        hex_color = rgb_to_hex(color_rgb)
        col.markdown(f"""
            <div class="color-box" style="background-color: {hex_color};"></div>
            <div class="hex-label">{hex_color}</div>
        """, unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="intro">
            👋 Selamat datang di <b>Dominant Color Picker</b>!<br>
            Upload gambar untuk melihat palet warna dominan yang cantik 🎨
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("---")
st.markdown('<div class="footer">© 2025 • Dibuat oleh Fadhila Latsa Tsabita</div>', unsafe_allow_html=True)
