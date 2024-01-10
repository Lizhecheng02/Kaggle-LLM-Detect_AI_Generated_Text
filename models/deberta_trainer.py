import torch
import pandas as pd
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

VER = 1


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base")

df = pd.read_parquet("../large dataset/data.parquet")
print("The shape of train data is:", df.shape)


def get_label(source):
    if source == "Human":
        return 0
    else:
        return 1


df["label"] = df["source"].apply(get_label)
print(df.head())


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.15
)
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

training_args = TrainingArguments(
    output_dir=f"../results_{VER}",
    num_train_epochs=2,
    learning_rate=4e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    report_to="none",
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=16,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=False,
    lr_scheduler_type="linear",
    weight_decay=0.01,
    save_total_limit=3
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=training_args.num_train_epochs *
    int(len(train_dataset) * 1.0 / training_args.per_device_train_batch_size /
        training_args.gradient_accumulation_steps),
    # num_training_steps=6750,
    power=1.0,
    lr_end=2.5e-6
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(f"../model_v{VER}")
