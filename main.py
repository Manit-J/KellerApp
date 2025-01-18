import streamlit as st
from PIL import Image

st.set_page_config(page_title="Keller App", page_icon="🧏", layout="wide")

# --- navigation sidebar ---
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ("Camera Translation", "Speech Translation", "Text Translation"))

