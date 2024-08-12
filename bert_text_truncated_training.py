from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from transformers import AdamW
from torch.optim import lr_scheduler
from tqdm import tqdm
import regex as re

def clean_text(text):
    # Remove all numbers and decimal values
    text = re.sub(r'\d+(\.\d+)?', '', text)
    
    # Replace multiple consecutive newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove special characters and symbols, keep only essential details
    # Allow letters, digits, spaces, and common punctuation
    text = re.sub(r'[^\w\s.,-]', '', text)
    
    # Remove extra spaces around punctuation and between words
    text = re.sub(r'\s+([.,-])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)


label_encoder = LabelEncoder()


text_dir = 'data/train_texts'
texts = []
main_target_labels = []

train_df = pd.read_csv('processed_train_data.csv', sep= '|')
train_df['target_label'] = label_encoder.fit_transform(train_df['target'])


for file in os.listdir(text_dir):
    if  len(train_df.loc[train_df['file_name'] == file.replace('.txt', '.pdf'), 'target_label'].values) == 0:
        pass
    else:
        with open(os.path.join(text_dir, file), 'r+') as f:
            texts.append(clean_text(f.read()))
            main_target_labels.append(train_df.loc[train_df['file_name'] == file.replace('.txt', '.pdf'), 'target_label'].values[0])
        print(file, train_df.loc[train_df['file_name'] == file.replace('.txt', '.pdf'), 'target_label'].values)



# Tokenize and prepare data

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# # Set parameters
max_length = 256
batch_size = 16

# # Create dataset and dataloaders
dataset = TextDataset(texts, main_target_labels, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# # Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# # Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

def train(model, dataloader, optimizer, scheduler, num_epochs):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    print("Training complete.")    
    model_save_path = 'bert_model_checkpoint_256_max_len_text_preprocessed.pth'
    torch.save(model.state_dict(), model_save_path)
    print('checkpoint saved!')

train(model, dataloader, optimizer, scheduler, 25)
