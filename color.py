import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Dominant Color Picker", layout="centered")

st.title("🎨 Dominant Color Picker")
st.markdown("Upload gambar dan dapatkan 5 warna dominan dari gambar tersebut.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

def get_dominant_colors(image, k=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    return colors

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Gambar yang diunggah", use_container_width=True)

    with st.spinner("Mengambil warna dominan..."):
        colors = get_dominant_colors(image, k=5)

    st.subheader("🎨 Palet Warna Dominan")

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        color_rgb = colors[idx]
        hex_color = rgb_to_hex(color_rgb)
        col.color_picker(f"Warna {idx + 1}", value=hex_color)
        col.markdown(f"<div style='text-align:center; font-size: 16px;'>{hex_color}</div>", unsafe_allow_html=True)
else:
    st.info("Silakan unggah gambar terlebih dahulu.")


