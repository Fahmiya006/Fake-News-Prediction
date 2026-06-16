import streamlit as st
import joblib

# Load model
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"

model = joblib.load(MODEL_PATH)

import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.sidebar.title("Project Info")

st.sidebar.write("""
Fake News Detection using:
- NLP
- TF-IDF
- Logistic Regression
- Streamlit
""")

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article and predict whether it is real or fake."
)
news_text = st.text_area(
    "Paste News Article Here",
    height=250
)

if st.button("Predict"):
    
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    
    else:
        prediction = model.predict([news_text])[0]

confidence = max(
    model.predict_proba([news_text])[0]
)

if prediction == 0:
    st.error("Fake News")
else:
    st.success("Real News")

st.metric(
    "Confidence Score",
    f"{confidence*100:.2f}%"
)