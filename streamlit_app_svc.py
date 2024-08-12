import streamlit as st
import joblib
import requests
import pdfplumber
import numpy as np
import re
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from io import BytesIO

# Load model and tokenizer
pipeline = joblib.load('ckpts/svc_text_classifier_chunked_text_model.pkl')
label_encoder = joblib.load('ckpts/svc_label_encoder.pkl')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Preprocessing functions
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\d+(\.\d+)?', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\w\s.,-]', ' ', text)
    text = re.sub(r'\s+([.,-])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[_\s]+', ' ', text)
    text = re.sub(r'[-\s]+', ' ', text)
    text = re.sub(r'[\.\,]+', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def chunk_text(document_text):
    chunks = []
    words = document_text.split()
    n = len(words) // 150
    if n < 1:
        n = 1
    for i in range(n):
        start = i * 150
        end = start + 200
        chunks.append(' '.join(words[start:end]))
    return chunks

def get_bert_cls_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return cls_embedding

def extract_text_from_pdf(pdf_content):
    text = ""
    with pdfplumber.open(BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Streamlit app
st.title("PDF Text Classification with SVC")

url = st.text_input("Enter PDF URL:")

if st.button("Classify PDF"):
    if url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                pdf_content = response.content
                text = extract_text_from_pdf(pdf_content)
                cleaned_text = clean_text(text)
                
                # Get embeddings for each chunk and predict
                embeddings = np.array([get_bert_cls_embedding(cleaned_text)])
                predictions = pipeline.predict(embeddings)
                
                # Decode predictions
                classes = label_encoder.classes_
                predicted_labels = [classes[pred] for pred in predictions]
                
                # Display results
                st.write(f"Predicted category: {', '.join(predicted_labels)}")
            else:
                st.error("Failed to download PDF. Please check the URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a URL.")
