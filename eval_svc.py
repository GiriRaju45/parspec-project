import numpy as np
import regex as re
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder


# Load SVC model and tokenizer
# pipeline = joblib.load('/Users/guest1/Desktop/parspec-assignment/svc_text_classifier_chunked_text_model.pkl')

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('bert-base-uncased')
# tokenizer = model.tokenize



def clean_text(text):
    # text = re.sub(r'\d+(\.\d+)?', ' ', text)
    # text = re.sub(r'\n+', '\n', text)
    # text = re.sub(r'[^\w\s.,-]', ' ', text)
    # text = re.sub(r'\s+([.,-])', r'\1', text)
    # text = re.sub(r'\s+', ' ', text)
    # text = re.sub(r'[_\s]+', ' ', text)
    # text = re.sub(r'[-\s]+', ' ', text)
    # text = re.sub(r'[\.\,]+', ' ', text)
    # text = re.sub(r'\b\w{1,2}\b', ' ', text)
    # text = text.lower()
    # text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # text = BAD_SYMBOLS_RE.sub('', text)
    # text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

# def get_bert_cls_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     outputs = bert_model(**inputs)
#     cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
#     return cls_embedding


# # Define a function for predictions using SVC
# def predict(texts, model):
#     embeddings = np.array([get_bert_cls_embedding(text) for text in texts])
#     predictions = model.predict(embeddings)
#     return predictions

# Load test data

# label_encoder = LabelEncoder()
# test_df = pd.read_csv('/Users/guest1/Desktop/parspec-assignment/processed_test_data.csv', sep='|')
# test_df['target_label'] = label_encoder.fit_transform(test_df['target'])
# text_dir = '/Users/guest1/Desktop/parspec-assignment/data/test_texts'
# test_texts = []
# target_labels = []

# for file in os.listdir(text_dir):
#     pdf_name = file.replace('.txt', '.pdf')
#     if not test_df.loc[test_df['file_name'] == pdf_name].empty:
#         label = test_df.loc[test_df['file_name'] == pdf_name, 'target_label'].values[0]
#         file_path = os.path.join(text_dir, file)
#         with open(file_path, 'r') as f:
#             cleaned_text = clean_text(f.read())
#             f.close()
#             test_texts.append(cleaned_text)
#             target_labels.append(label)

# # Predict
# predictions = predict(test_texts, pipeline)

# # Compute metrics
# accuracy = accuracy_score(target_labels, predictions)
# precision = precision_score(target_labels, predictions, average='weighted')
# recall = recall_score(target_labels, predictions, average='weighted')
# f1 = f1_score(target_labels, predictions, average='weighted')

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")

# # # Map prediction IDs to labels
# # classes = label_encoder.classes_
# # id2label = {idx: label for idx, label in enumerate(classes)}

# # # Print results
# # for text, label in zip(test_texts[:10], target_labels[:10]):
# #     predicted_label = id2label[predictions[text]]
# #     print(f'Text: {text[:100]}... \nActual Label: {id2label[label]} \nPredicted Label: {predicted_label}\n')
