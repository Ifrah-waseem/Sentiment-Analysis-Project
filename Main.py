import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("Sentiment Analysis App")


# Cache the model so it loads ONCE, not on every rerun/button click.
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english"
    )

with st.spinner("Loading model..."):
    sentiment_analysis = load_model()

# --- Sidebar ---
st.sidebar.title("Menu")

st.sidebar.subheader("About")
st.sidebar.write(
    "This app uses the Hugging Face Transformers library "
    "to perform sentiment analysis on text data. "
    "You can either enter text directly or upload a CSV "
    "file for batch analysis."
)

st.sidebar.subheader("Features")
st.sidebar.markdown(
    "- Single text analysis\n"
    "- CSV batch analysis\n"
    "- Confidence score\n"
    "- Download results"
)

st.sidebar.subheader("Model")
st.sidebar.write("RoBERTa — Siebert Sentiment")
st.sidebar.caption("Note: binary POSITIVE/NEGATIVE only — no neutral class, and it can misread sarcasm.")

# --- Single text analysis ---
st.header("Single Text Analysis")
text_input = st.text_area("Enter text for sentiment analysis")

if st.button("Analyze Text"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_analysis(text_input[:512])  # guard against very long input

        label = result[0]["label"]
        score = result[0]["score"]

        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {score:.1%}")
    else:
        st.warning("Please enter some text for analysis.")

# --- Batch analysis ---
st.header("Batch Analysis (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Your uploaded data:")
    st.dataframe(df, use_container_width=True)

    # Let the user pick the text column instead of hardcoding "review"
    text_column = st.selectbox("Select the column containing text to analyze", df.columns)

    if st.button("Analyze CSV"):
        results = []
        progress_bar = st.progress(0, text="Analyzing...")
        total_rows = len(df)

        for i, text in enumerate(df[text_column]):
            if pd.isna(text) or not str(text).strip():
                results.append({"Text": text, "Sentiment": "SKIPPED", "Confidence": None})
            else:
                try:
                    result = sentiment_analysis(str(text)[:512])
                    results.append({
                        "Text": text,
                        "Sentiment": result[0]["label"],
                        "Confidence": round(result[0]["score"], 4),
                    })
                except Exception as e:
                    results.append({"Text": text, "Sentiment": "ERROR", "Confidence": None})

            progress_bar.progress((i + 1) / total_rows, text=f"Analyzing... {i + 1}/{total_rows}")

        progress_bar.empty()

        # Store in session_state so results survive reruns (e.g. sidebar interaction)
        st.session_state["result_df"] = pd.DataFrame(results)

# --- Show results if they exist in session state ---
if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]

    st.subheader("Statistics")
    total_reviews = len(result_df)
    positive_reviews = (result_df["Sentiment"] == "POSITIVE").sum()
    negative_reviews = (result_df["Sentiment"] == "NEGATIVE").sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", total_reviews)
    with col2:
        st.metric("Positive", positive_reviews)
    with col3:
        st.metric("Negative", negative_reviews)

    st.bar_chart(result_df["Sentiment"].value_counts())

    st.subheader("Sentiment Results")
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )