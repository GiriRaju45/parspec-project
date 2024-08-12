import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from transformers import AdamW
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Load the checkpoint
checkpoint_path = '/Users/guest1/Desktop/parspec-assignment/bert_model_checkpoint_512_max_len_text_chunked.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()


def predict(texts, model, tokenizer, max_length):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    predictions = []
    
    # Tokenize and prepare inputs
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


# Example texts for inference
import re
from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Replace numbers with optional decimal part with a space
    text = re.sub(r'\d+(\.\d+)?', ' ', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace any non-word characters (except whitespace, periods, commas, and dashes) with a space
    text = re.sub(r'[^\w\s.,-]', ' ', text)
    
    # Remove spaces before punctuation marks (., -)
    text = re.sub(r'\s+([.,-])', r'\1', text)
    
    # Replace multiple spaces with a single space and trim leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace underscores and spaces with a single space
    text = re.sub(r'[_\s]+', ' ', text)

    # text = re.sub(r'[#+\*]', ' ', text)
    
    # Replace dashes and spaces with a single space
    text = re.sub(r'[-\s]+', ' ', text)

    text = re.sub(r'[\.\,]+', ' ', text )

    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#     text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    # Remove extra spaces that may be left behind
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    cleaned_text = cleaned_text.replace('#', '')


    
    return cleaned_text.strip()

test_texts = []
target_labels = []

label_encoder = LabelEncoder()
test_df = pd.read_csv('processed_test_data.csv', sep='|')
test_df['target_label'] = label_encoder.fit_transform(test_df['target'])

text_dir = '/Users/guest1/Desktop/parspec-assignment/data/train_texts2'
for file in os.listdir(text_dir):
    pdf_name = file.replace('.txt', '.pdf')
    if not test_df.loc[test_df['file_name'] == pdf_name].empty:
        label = test_df.loc[test_df['file_name'] == pdf_name, 'target_label'].values[0]
        file_path = os.path.join(text_dir, file)
        with open(file_path, 'r') as f:
            cleaned_text = clean_text(f.read())
            test_texts.append([file, cleaned_text])
            target_labels.append(label)

texts_to_predict = [clean_text(text) for file, text in test_texts]

# Get predictions
predictions = predict(texts_to_predict, model, tokenizer, 512)

print(predictions)
print((file,text) for file, text in test_texts[-3:])

classes = ['cable','fuses', 'lighting', 'others']

label2id = {label: idx for idx, label in enumerate(classes)}
id2label = {idx: label for idx, label in enumerate(classes)}

for pred, (file,text) in zip(predictions, test_texts[-3:]):
    print(f' The file - {file} \n belongs to the category: {id2label[pred]} \n\n')

accuracy = accuracy_score(target_labels, predictions)
precision = precision_score(target_labels, predictions, average='weighted')
recall = recall_score(target_labels, predictions, average='weighted')
f1 = f1_score(target_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")