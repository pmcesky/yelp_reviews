import sys
import gzip
import shutil
import time
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np
import mlflow

# > DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased , runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.

# ## Fine-tuning a BERT model in PyTorch


# **General Settings**

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 100
BATCH_SIZE = 64


# ### Loading the Yelp user review dataset

# Check that the dataset looks okay:

df = pd.read_csv('./yelp_review_first_130K_with_sentiment.csv')

# **Split Dataset into Train/Validation/Test**
# Train 100K, Valid 10K, Test 20K

train_texts = df.iloc[:100000]['text'].values
train_labels = df.iloc[:100000]['sentiment'].values

valid_texts = df.iloc[100000:110000]['text'].values
valid_labels = df.iloc[100000:110000]['sentiment'].values

test_texts = df.iloc[110000:]['text'].values
test_labels = df.iloc[110000:]['sentiment'].values


# ## Tokenizing the dataset

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

# **Dataset Class and Loaders**

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = YelpDataset(train_encodings, train_labels)
valid_dataset = YelpDataset(valid_encodings, valid_labels)
test_dataset = YelpDataset(test_encodings, test_labels)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ## Loading and fine-tuning a pre-trained BERT model

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


# **Train Model -- Manual Training Loop**

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        
        for batch_idx, batch in enumerate(data_loader):
        
        ### Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
        
        return correct_pred.float()/num_examples * 100


# Train and record training with MLFLOW
mlflow.set_experiment('TEST')
with mlflow.start_run():
    
    mlflow.log_param('Batch_size', BATCH_SIZE)
    mlflow.log_param('Num_epochs', NUM_EPOCHS)
    mlflow.log_param('Torch_random_seed', RANDOM_SEED)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        
        model.train()
        epoch_loss = []
        for batch_idx, batch in enumerate(train_loader):
            
            ### Prepare data
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            ### Forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs['loss'], outputs['logits']
            epoch_loss.append(loss)
            
            ### Backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            ### Logging
            if not batch_idx % 250:
                print (f'Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} | '
                    f'Batch {batch_idx:04d}/{len(train_loader):04d} | '
                    f'Loss: {loss:.4f} | '
                    f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
                
        model.eval()

        # Log with MLFLOW
        with torch.set_grad_enabled(False):
            train_accuracy = compute_accuracy(model, train_loader, DEVICE)
            valid_accuracy = compute_accuracy(model, valid_loader, DEVICE)
            test_accuracy = compute_accuracy(model, test_loader, DEVICE)

            mlflow.log_metric(key='Train Loss', value=torch.tensor(epoch_loss).float().mean(), step = epoch+1)
            mlflow.log_metric(key='Train accuracy', value=train_accuracy, step = epoch+1)
            mlflow.log_metric(key='Valid accuracy', value=valid_accuracy, step = epoch+1)
            mlflow.log_metric(key='Test accuracy', value=test_accuracy, step = epoch+1)
            print(f'Training accuracy: {train_accuracy:.2f}%'
                f'\nValid accuracy: {valid_accuracy:.2f}%'
                f'\nTest accuracy: {test_accuracy:.2f}%')

        elapsed_time = (time.time() - start_time)/60
        mlflow.log_metric(key='Elapsed_time', value=elapsed_time, step = epoch+1)            
        print(f'Time elapsed: {elapsed_time:.2f} min')
        print()
        if not (epoch+1)%10:
            mlflow.pytorch.log_model(model, 'model')
            ############ Or log the model the following way
            # checkpoint={'model_dict':model.state_dict(), 'optimizer':optim.state_dict()}
            # torch.save(checkpoint,'./models/model'+str(batch_idx)+'.pth.tar')
            # mlflow.log_artifact('./models/model'+str(batch_idx)+'.pth.tar', 'models')


    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
    print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')
