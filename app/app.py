import streamlit as st
import joblib
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -----------------------------
# Load model (Pipeline)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# UI Sidebar
# -----------------------------
st.sidebar.title("Project Info")
st.sidebar.write("""
Fake News Detection System

- NLP (TF-IDF)
- Machine Learning (Logistic Regression)
- Streamlit Deployment
""")

# -----------------------------
# Main UI
# -----------------------------
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

news_text = st.text_area("Paste News Article Here", height=200)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    if len(news_text.strip()) == 0:
        st.warning("Please enter a news article.")
    else:

        prediction = model.predict([news_text])[0]
        confidence = max(model.predict_proba([news_text])[0])

        if prediction == 0:
            st.error("⚠️ Fake News")
        else:
            st.success("✅ Real News")

        st.metric("Confidence Score", f"{confidence*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("⚠️ This model is for educational purposes only and does not guarantee factual verification.")