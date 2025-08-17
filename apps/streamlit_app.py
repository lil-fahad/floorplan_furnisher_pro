
import io
import streamlit as st
from PIL import Image
from furniture_ai.infer.furnish import run_furnish

st.set_page_config(page_title="Floorplan Furnisher Pro", page_icon="🧭", layout="wide")
st.title("🧭 Floorplan Furnisher Pro")
st.caption("ارفع مخططك، نحلّل ونفرش غرفك تلقائيًا (MVP).")

weights_path = st.sidebar.text_input("مسار أوزان الـ Segmenter", value="models/segmenter/best.pt")
file = st.file_uploader("ارفع مخطط (PNG/JPG)", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="المخطط", use_column_width=True)
    with st.spinner("تحليل + تأثيث..."):
        out = run_furnish(img, weights_path=weights_path)
    st.success("تم!")
    st.subheader("Overlay")
    st.image(io.BytesIO(out["overlay_png"]), use_column_width=True)
    st.subheader("Layout JSON")
    st.json(out["layout"])
else:
    st.info("حمّل صورة مخطط للبدء.")
