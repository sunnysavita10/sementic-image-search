import io
import requests
import streamlit as st
from PIL import Image

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Semantic Image Search", layout="wide")
st.title("🔍 Semantic Image Search (CLIP + Qdrant + LLM)")

tab1, tab2 = st.tabs(["Search by Text", "Search by Image"])


with tab1:
    query = st.text_input("Enter your search query")
    k = st.slider("Top-K results", 1, 10, 5)

    if st.button("Search Images"):
        if not query:
            st.warning("Please enter a query.")
        else:
            params = {"q": query, "k": k}
            res = requests.get(f"{API_BASE}/search-text", params=params)
            data = res.json()

            st.write("Translated Query:", data.get("translated"))

            cols = st.columns(3)
            for idx, item in enumerate(data.get("results", [])):
                with cols[idx % 3]:
                    st.image(item["path"], caption=f"{item['filename']} (score={item['score']:.3f})")


with tab2:
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    k2 = st.slider("Top-K results (image query)", 1, 10, 5, key="k_image")

    if upload:
        img = Image.open(upload)
        st.image(img, caption="Query Image", use_container_width=True)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        if st.button("Find Similar Images"):
            files = {"file": ("query.png", buf, "image/png")}
            res = requests.post(f"{API_BASE}/search-image", files=files, params={"k": k2})
            data = res.json()

            cols = st.columns(3)
            for idx, item in enumerate(data.get("results", [])):
                with cols[idx % 3]:
                    st.image(item["path"], caption=f"{item['filename']} (score={item['score']:.3f})")
