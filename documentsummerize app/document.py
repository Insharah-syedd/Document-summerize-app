import streamlit as st
import easyocr
import fitz  # PyMuPDF
import numpy as np
import torch
import io
import base64
import os
from PIL import Image
from transformers import pipeline

# --- CSS Loading Function ---
def load_css(file_name):
    """Reads the style.css file and applies it to the Streamlit app."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"{file_name} not found. Please ensure it is in the same folder.")

# --- Model Loading (Optimized for English) ---
@st.cache_resource
def load_summarizer():
    # Using BART model for high-quality English summarization
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# --- OCR Reader Loading ---
@st.cache_resource
def load_ocr(lang):
    return easyocr.Reader([lang])

# --- Logic Functions ---

def extract_text_from_pdf(pdf_path, is_scanned=True):
    doc = fitz.open(pdf_path)
    text = ""
    
    if not is_scanned:
        # Fast extraction for standard PDFs
        for page in doc:
            text += page.get_text()
    else:
        # OCR extraction for scanned/image PDFs
        reader = load_ocr('en')
        for page in doc:
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
            result = reader.readtext(img_array)
            text += ' '.join([item[1] for item in result]) + '\n'
    return text.strip()

def summarize_text(text):
    if not text or len(text.split()) < 20:
        return "The text is too short to generate a meaningful summary."
    
    # BART model input limit handling
    input_text = text[:3000] 
    summary = summarizer(input_text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Main Streamlit UI ---
# Set page config before loading CSS
st.set_page_config(layout="wide", page_title="AI Document Summarizer")

def main():
    # Load your custom 3D CSS file
    load_css('style.css')
    
    st.title("📄 AI Document Summarization App")
    
    # Create data directory for file handling
    if not os.path.exists("data"):
        os.makedirs("data")

    option = st.selectbox("Select Input Type", ('PDF', 'Plain Text', 'Image'))

    if option == 'PDF':
        uploaded_file = st.file_uploader("Upload PDF File", type=['pdf'])
        is_scanned = st.checkbox("Is this a scanned PDF (image-based)?", value=False)

        if uploaded_file and st.button("Summarize"):
            filepath = os.path.join("data", uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("Original Document View")
                displayPDF(filepath)
            
            with col2:
                with st.spinner('AI is processing your document...'):
                    extracted_text = extract_text_from_pdf(filepath, is_scanned)
                    summary = summarize_text(extracted_text)
                    
                    st.subheader("Extracted Content Preview:")
                    st.text_area("", extracted_text[:1000] + "...", height=200)
                    
                    st.subheader("✨ AI Summary:")
                    st.success(summary)

    elif option == 'Plain Text':
        user_text = st.text_area("Paste your text here:", height=300)
        if st.button("Summarize Text") and user_text:
            with st.spinner('Summarizing...'):
                summary = summarize_text(user_text)
                st.success(summary)

    elif option == 'Image':
        uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_img and st.button("Extract & Summarize"):
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded Image Preview", width=400)
            
            with st.spinner('Extracting text from image...'):
                reader = load_ocr('en')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                result = reader.readtext(img_byte_arr.getvalue())
                extracted_text = ' '.join([item[1] for item in result])
                
                summary = summarize_text(extracted_text)
                st.subheader("AI Summary:")
                st.success(summary)

if __name__ == "__main__":
    main()
    