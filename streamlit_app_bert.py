import streamlit as st
import requests
import os
import pdfplumber
import re
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define paths and load models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
checkpoint_path = 'ckpts/bert_model_checkpoint_512_max_len_text_chunked.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Helper functions
def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    
    text = re.sub(r'\d+(\.\d+)?', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\w\s.,-]', ' ', text)
    text = re.sub(r'\s+([.,-])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[_\s]+', ' ', text)
    text = re.sub(r'[-\s]+', ' ', text)
    text = re.sub(r'[\.\,]+', ' ', text )
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text.strip()

def predict(texts, model, tokenizer, max_length):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    predictions = []

    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            predictions.append(prediction)

    return predictions

def extract_text_from_pdf(pdf_content):
    text = ""
    with pdfplumber.open(BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Streamlit UI
st.title("BERT Model for PDF Classification")

url = st.text_input("Enter PDF URL:")

if st.button("Classify PDF"):
    if url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                pdf_content = response.content
                text = extract_text_from_pdf(pdf_content)
                cleaned_text = clean_text(text)
                texts_to_predict = [cleaned_text]

                predictions = predict(texts_to_predict, model, tokenizer, 512)
                classes = ['cable', 'fuses', 'lighting', 'others']
                id2label = {idx: label for idx, label in enumerate(classes)}

                if predictions:
                    pred_label = id2label[predictions[0]]
                    st.write(f"The PDF belongs to the category: {pred_label}")
            else:
                st.error("Failed to download PDF. Please check the URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a URL.")
