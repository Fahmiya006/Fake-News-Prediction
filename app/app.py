import streamlit as st
import joblib
from pathlib import Path
import numpy as np

st.set_page_config(
    page_title="Advanced Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "fake_news_model.pkl")

st.title("🧠 Advanced Fake News Detection System")

st.write("Paste a news article below:")

news_text = st.text_area("News Input", height=250)

if st.button("Analyze"):

    if len(news_text.strip()) < 30:
        st.warning("Please enter a full news article.")
    else:

        prob = model.predict_proba([news_text])[0]
        prediction = np.argmax(prob)

        fake_prob = prob[0]
        real_prob = prob[1]

        st.subheader("Prediction Result")

        if abs(fake_prob - real_prob) < 0.15:
            st.warning("⚠️ Uncertain News")
        elif prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.subheader("Confidence Scores")

        st.progress(float(real_prob))
        st.write(f"Real News Probability: {real_prob*100:.2f}%")

        st.progress(float(fake_prob))
        st.write(f"Fake News Probability: {fake_prob*100:.2f}%")