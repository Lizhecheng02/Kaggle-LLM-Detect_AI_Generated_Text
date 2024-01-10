import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

VER = 1


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


tokenizer = DebertaV2Tokenizer.from_pretrained(
    "microsoft/deberta-v3-base"
)
model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base"
)

df = pd.read_parquet("../large_dataset/data.parquet")
print("The shape of train data is:", df.shape)


def get_label(source):
    if source == "Human":
        return 0
    else:
        return 1


df["label"] = df["source"].apply(get_label)
print(df.head())
df["label"].value_counts().plot(
    kind="bar", title="Distribution of Labels"
)
plt.show()


class CustomDataset(Dataset):
    def __init__(self, encodings, labels, tokenizer):
        self.encodings = encodings
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
    df["text"], df["label"],
    test_size=0.15, random_state=2024
)

train_texts = train_texts.tolist()
tokenized_train_texts = []
for train_text in tqdm(train_texts, desc="Tokenizing Train Texts"):
    tokenized_train_text = tokenizer(
        train_text,
        padding="max_length",
        max_length=512,
        truncation=True
    )
    tokenized_train_texts.append(tokenized_train_text)

with open("../tokenized_data/train_tokenized_data.json", "w") as file:
    json.dump(tokenized_train_texts, file)

with open("../tokenized_data/train_tokenized_data.json", "r") as file:
    tokenized_train_texts = json.load(file)

val_texts = val_texts.tolist()
tokenized_val_texts = []
for val_text in tqdm(val_texts, desc="Tokenizing Val Texts"):
    tokenized_val_text = tokenizer(
        val_text,
        padding="max_length",
        max_length=512,
        truncation=True
    )
    tokenized_val_texts.append(tokenized_val_text)

with open("../tokenized_data/val_tokenized_data.json", "w") as file:
    json.dump(tokenized_val_texts, file)

with open("../tokenized_data/val_tokenized_data.json", "r") as file:
    tokenized_val_texts = json.load(file)

train_dataset = CustomDataset(
    tokenized_train_texts,
    train_labels.tolist(),
    tokenizer
)
val_dataset = CustomDataset(
    tokenized_val_texts,
    val_labels.tolist(),
    tokenizer
)

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
    int(len(train_texts) * 1.0 / training_args.per_device_train_batch_size /
        training_args.gradient_accumulation_steps),
    power=1.0,
    lr_end=2.5e-6
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

trainer.train()
trainer.save_model(f"../model_v{VER}")
