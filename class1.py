!pip install transformers datasets scikit-learn

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from datetime import datetime



# Custom Dataset for HUPD Data
class HUPDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        title = record.get("title", "")
        abstract = record.get("abstract", "")
        label = record.get("label", 0)  # Default to 0 if label is missing

        # Combine title and abstract
        combined_text = f"Title: {title}\nAbstract: {abstract}"

        # Tokenize the text
        encoded = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Fine-tune the Pre-trained BERT Model
def fine_tune_model(model, train_dataloader, val_dataloader, epochs=3, lr=5e-5, device="cuda"):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

        # Validate after each epoch
        validate_model(model, val_dataloader, device)


# Validate the Model
def validate_model(model, val_dataloader, device="cuda"):
    model.eval()
    model.to(device)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")


# Main Function to Run the Workflow
def main():
    # Configuration
    model_name = "anferico/bert-for-patents"
    batch_size = 8
    max_length = 128
    epochs = 3
    learning_rate = 5e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Specify the date ranges as required by the HUPD dataset API
    train_filing_start_date = "2016-01-01"
    train_filing_end_date = "2016-01-20"
    val_filing_start_date = "2016-01-20"
    val_filing_end_date = "2016-01-31"

    # Load the HUPD dataset with the sample data for debugging
    print("Loading HUPD dataset...")
    try:
        dataset = load_dataset(
            "HUPD/hupd",
            "sample",  # Use sample for debugging
            train_filing_start_date=train_filing_start_date,
            train_filing_end_date=train_filing_end_date,
            val_filing_start_date=val_filing_start_date,
            val_filing_end_date=val_filing_end_date,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Perform a uniform split for training and validation
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Preprocess datasets
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_data = HUPDDataset(train_dataset, tokenizer, max_length)
    val_data = HUPDDataset(val_dataset, tokenizer, max_length)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Load the pre-trained model
    print("Loading the model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tune_model(model, train_dataloader, val_dataloader, epochs, learning_rate, device)


if __name__ == "__main__":
    main()




