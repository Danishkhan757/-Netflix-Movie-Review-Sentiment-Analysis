import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Netflix Movie Analyzer", layout="wide")

st.title("🎬 Netflix Movie Review Analyzer")

@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization")
    return sentiment_analyzer, summarizer

sentiment_analyzer, summarizer = load_models()

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Review" in df.columns:
        review = st.text_area("Enter Review")

        if st.button("Analyze Sentiment"):
            result = sentiment_analyzer(review)
            st.write(result)

        if st.button("Summarize Review"):
            summary = summarizer(
                review,
                max_length=50,
                min_length=10,
                do_sample=False
            )
            st.write(summary[0]["summary_text"])
    else:
        st.error("CSV must contain a 'Review' column.")