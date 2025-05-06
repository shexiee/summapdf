# PDF Summarizer

A web application built with Streamlit that summarizes PDF documents using Hugging Face's BART-large-CNN model.

## Features

- User-friendly web interface for uploading PDF documents
- Interactive controls for customizing summary length
- Text extraction from PDF files
- Summarization using BART-large-CNN model
- Progress tracking for multi-chunk documents
- Download option for generated summaries

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/shexiee/summapdf.git
   cd summapdf
   ```

2. Install the required dependencies:

   Option 1: Using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

   Option 2: Manual installation:
   ```
   pip install torch
   pip install transformers
   pip install PyPDF2
   pip install streamlit
   ```

   Note: The first time you run the app, it will download the BART model (approximately 1.6GB).
   These files will be stored in a cache directory and won't be pushed to GitHub.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the interface to:
   - Upload a PDF file
   - Adjust summary parameters using the sidebar controls
   - Generate and download the summary

## Deployment

You can deploy this app to Render for free:

1. Push your code to a GitHub repository
2. Visit [render.com](https://render.com)
3. Connect your GitHub account and select this repository
4. Deploy the app

## How it Works

1. The app extracts text from the uploaded PDF using PyPDF2
2. The text is split into chunks to accommodate the BART model's maximum input length
3. Each chunk is summarized using the BART-large-CNN model
4. The summaries are combined to form a complete summary of the document
5. Users can download the generated summary as a text file

## Requirements

- Python 3.7+
- PyPDF2
- Transformers
- PyTorch
- Streamlit

## Note

The first time you run the application, it will download the BART-large-CNN model (approximately 1.6GB), which might take some time depending on your internet connection. The model will be cached for subsequent runs.
