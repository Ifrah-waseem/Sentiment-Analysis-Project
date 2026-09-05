# Sentiment Analysis App

A simple Streamlit app that performs sentiment analysis on text using a Hugging Face Transformers model. Supports both single-text analysis and batch analysis via CSV upload.

## Features

- **Single text analysis** — type or paste text and get an instant sentiment label with confidence score
- **Batch analysis** — upload a CSV file and analyze an entire column of text at once
- **Progress tracking** — live progress bar during batch analysis
- **Statistics dashboard** — total, positive, and negative counts with a bar chart
- **Downloadable results** — export batch results as a CSV file

## Model

This app uses [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english), a RoBERTa-based model fine-tuned for English sentiment classification.

**Note:** The model outputs only `POSITIVE` or `NEGATIVE` (no neutral class), and like most sentiment models, it can struggle with sarcasm or mixed sentiment.

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   ```
   - Windows (cmd): `.venv\Scripts\activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Mac/Linux: `source .venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the app with:

```
streamlit run Main.py
```

Then open the local URL Streamlit gives you (usually `http://localhost:8501`).

### Single text analysis
Type or paste text into the text box and click **Analyze Text**.

### Batch analysis
1. Upload a CSV file.
2. Select the column that contains the text to analyze.
3. Click **Analyze CSV**.
4. View the statistics and results table, then download the results as a CSV if needed.

## Project Structure

```
├── Main.py              # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Limitations

- Text longer than 512 tokens is truncated before analysis.
- The model only classifies POSITIVE/NEGATIVE — there is no neutral category.
- Batch analysis speed depends on your machine's CPU/GPU and the number of rows in the CSV.

## License

This project is for learning/personal use. The underlying model is subject to its own license on [Hugging Face](https://huggingface.co/siebert/sentiment-roberta-large-english).
