import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from transformers import pipeline

# Set page configuration
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

@st.cache_resource
def load_summarizer():
    """Load and cache the summarization model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def summarize_text(text, max_length=130, min_length=30):
    """Summarize text using the BART-large-CNN model."""
    try:
        summarizer = load_summarizer()
        
        # BART has a maximum input length, so we need to chunk the text
        max_chunk_length = 1024  # BART's maximum input length
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        total_chunks = len(chunks)
        
        # Create a progress bar
        if total_chunks > 1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only summarize meaningful chunks
                if total_chunks > 1:
                    status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
                    progress_bar.progress((i+1)/total_chunks)
                
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        # Clear the progress indicators if they were created
        if total_chunks > 1:
            status_text.empty()
            progress_bar.empty()
            
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return None

def main():
    st.title("PDF Summarizer")
    st.write("Upload a PDF document and get a concise summary using BART-large-CNN")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Summary Settings")
        max_length = st.slider("Maximum summary length", 50, 500, 130, 10)
        min_length = st.slider("Minimum summary length", 10, 200, 30, 5)
        
        st.write("---")
        st.write("Using model: facebook/bart-large-cnn")
        
    if uploaded_file is not None:
        # Display file details
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write("**File Details:**")
        st.json(file_details)
        
        # Create a button to process the PDF
        if st.button("Generate Summary"):
            with st.spinner("Extracting text from PDF..."):
                # Create a temporary file to save the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                # Extract text from the PDF
                text = extract_text_from_pdf(temp_path)
                
                # Remove the temporary file
                os.unlink(temp_path)
                
                if text:
                    st.success(f"Text extracted: {len(text)} characters")
                    
                    # Show a sample of the extracted text
                    with st.expander("Show extracted text sample"):
                        st.text(text[:500] + ("..." if len(text) > 500 else ""))
                    
                    # Summarize the text
                    with st.spinner("Generating summary... This may take a while for large documents"):
                        summary = summarize_text(text, max_length=max_length, min_length=min_length)
                    
                    if summary:
                        st.subheader("Summary")
                        st.write(summary)
                        
                        # Add download button for the summary
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name=f"{uploaded_file.name.split('.')[0]}_summary.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to generate summary.")
                else:
                    st.error("Failed to extract text from PDF.")
    
    # Add some information at the bottom
    st.markdown("---")
    st.markdown(
        """
        This app uses Hugging Face's BART-large-CNN model to generate summaries. 
        The first run may take some time as the model needs to be downloaded.
        """
    )

if __name__ == "__main__":
    main()