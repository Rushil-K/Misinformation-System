import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# -------------------- Load Dataset --------------------
df = pd.read_csv("liar_dataset/train.tsv", sep='\t', header=None)
df.columns = ["ID", "Label", "Statement", "Subject", "Speaker", "Job", "State", "Party", "Context"]
LABELS = ["False", "Half-True", "Mostly-True", "True", "Barely-True", "Pants-on-Fire"]
df["Label"] = df["Label"].astype("category").cat.codes

# -------------------- Dataset Class --------------------
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

# -------------------- Train Model --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_texts, test_texts, train_labels, test_labels = train_test_split(df["Statement"].tolist(), df["Label"].tolist(), test_size=0.2, random_state=42)

train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = HybridBERTLSTM().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss {loss.item()}")

torch.save(model.state_dict(), "hybrid_bert_lstm.pth")
print("Model training complete. ✅")
