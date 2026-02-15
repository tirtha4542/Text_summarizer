import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from torch.optim import AdamW

# 1. Setup Device & Data
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dataset = load_dataset("knkarthick/samsum")

model_id = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_id)
pretrain_model = BartForConditionalGeneration.from_pretrained(model_id)


# 2. Dataset Class
class SummarizationDataset(Dataset):
    def __init__(self, hf_ds, tokenizer, max_input_length=1024, max_target_length=128):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        example = self.hf_ds[idx]

        # Tokenize Inputs
        inputs = self.tokenizer(
            example['dialogue'],
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        # Tokenize Targets
        targets = self.tokenizer(
            example["summary"],
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        labels = targets["input_ids"].squeeze(0)

        # IMPORTANT: Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }


# 3. Data Module
class SummarizationDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_set = SummarizationDataset(dataset['train'], self.tokenizer)
        self.val_set = SummarizationDataset(dataset['validation'], self.tokenizer)
        self.test_set = SummarizationDataset(dataset['test'], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


# 4. Lightning Model Module
class SummarizationModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


# 5. Execution
if __name__ == "__main__":
    # Initialize components
    data_module = SummarizationDataModule(tokenizer=tokenizer, batch_size=4)
    lightning_model = SummarizationModel(pretrain_model)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    # Train
    trainer.fit(lightning_model, datamodule=data_module)