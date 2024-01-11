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
    AutoConfig
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")


class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=0.0001):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data,
                            self.backup_eps[name][0]
                        ),
                        self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class CustomTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        awp_lr=0,
        awp_eps=0,
        awp_start_epoch=0
    ):

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.awp_start_epoch = awp_start_epoch

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # AWP
        if self.awp_lr != 0 and self.state.epoch >= self.awp_start_epoch:
            self.awp = AWP(model, adv_lr=self.awp_lr, adv_eps=self.awp_eps)
            self.awp._save()
            self.awp._attack_step()
            with self.compute_loss_context_manager():
                awp_loss = self.compute_loss(self.awp.model, inputs)

            if self.args.n_gpu > 1:
                awp_loss = awp_loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(awp_loss).backward()
            elif self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()

        return loss.detach() / self.args.gradient_accumulation_steps


def train(args):

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
        args.model_name
    )
    model = DebertaV2ForSequenceClassification.from_pretrained(
        args.model_name
    )

    config = AutoConfig.from_pretrained(args.model_name)
    config.hidden_dropout_prob = args.dropout_rate
    config.attention_probs_dropout_prob = args.dropout_rate
    model = DebertaV2ForSequenceClassification.from_pretrained(
        args.model_name,
        config=config
    )

    df = pd.read_parquet("../large_dataset/data.parquet")
    num_samples = min(len(df), args.num_samples)
    df = df.sample(num_samples)
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
        test_size=args.test_size, random_state=args.random_state
    )

    if not args.is_load_from_disk:
        train_texts = train_texts.tolist()
        tokenized_train_texts = []
        for train_text in tqdm(train_texts, desc="Tokenizing Train Texts"):
            tokenized_train_text = tokenizer(
                train_text,
                padding="max_length",
                max_length=args.max_length,
                truncation=True
            )
            serialized_data = {key: value for key,
                               value in tokenized_train_text.items()}
            tokenized_train_texts.append(serialized_data)

        with open("../tokenized_data/train_tokenized_data.json", "w") as file:
            json.dump(tokenized_train_texts, file)
    else:
        with open("../tokenized_data/train_tokenized_data.json", "r") as file:
            tokenized_train_texts = json.load(file)

    if not args.is_load_from_disk:
        val_texts = val_texts.tolist()
        tokenized_val_texts = []
        for val_text in tqdm(val_texts, desc="Tokenizing Val Texts"):
            tokenized_val_text = tokenizer(
                val_text,
                padding="max_length",
                max_length=args.max_length,
                truncation=True
            )
            serialized_data = {key: value for key,
                               value in tokenized_val_text.items()}
            tokenized_val_texts.append(serialized_data)

        with open("../tokenized_data/val_tokenized_data.json", "w") as file:
            json.dump(tokenized_val_texts, file)
    else:
        with open("../tokenized_data/val_tokenized_data.json", "r") as file:
            tokenized_val_texts = json.load(file)

    train_encodings = {key: [] for key in tokenized_train_texts[0].keys()}
    for entry in tokenized_train_texts:
        for key in train_encodings.keys():
            train_encodings[key].append(entry[key])

    for key in train_encodings.keys():
        train_encodings[key] = torch.tensor(train_encodings[key])

    val_encodings = {key: [] for key in tokenized_val_texts[0].keys()}
    for entry in tokenized_val_texts:
        for key in val_encodings.keys():
            val_encodings[key].append(entry[key])

    for key in val_encodings.keys():
        val_encodings[key] = torch.tensor(val_encodings[key])

    train_dataset = CustomDataset(
        train_encodings,
        train_labels.tolist(),
        tokenizer
    )
    val_dataset = CustomDataset(
        val_encodings,
        val_labels.tolist(),
        tokenizer
    )

    training_args = TrainingArguments(
        output_dir=f"../results_{args.version}",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        report_to="none",
        overwrite_output_dir=args.overwrite_output_dir,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit
    )

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=training_args.num_train_epochs *
        int(len(train_texts) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps),
        power=args.power,
        lr_end=args.lr_end
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        awp_lr=args.awp_lr,
        awp_eps=args.awp_eps,
        awp_start_epoch=args.awp_start_epoch
    )

    trainer.train()
    trainer.save_model(f"../model_v{args.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a Transformers Model on a Text Classification Task"
    )
    parser.add_argument("--version", default=1, type=int)
    parser.add_argument("--is_load_from_disk", default=False, type=bool)
    parser.add_argument(
        "--model_name", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--num_samples", default=1000000, type=int)
    parser.add_argument("--test_size", default=0.10, type=float)
    parser.add_argument("--random_state", default=2024, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--learning_rate", default=4e-6, type=float)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
    parser.add_argument("--fp16", default=True, type=bool)
    parser.add_argument("--overwrite_output_dir", default=True, type=bool)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--load_best_model_at_end", default=True, type=bool)
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--save_total_limit", default=3, type=int)
    parser.add_argument("--num_warmup_steps", default=100, type=int)
    parser.add_argument("--power", default=1.0, type=float)
    parser.add_argument("--lr_end", default=2e-6, type=float)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--awp_lr", default=0.1, type=float)
    parser.add_argument("--awp_eps", default=1e-4, type=float)
    parser.add_argument("--awp_start_epoch", default=0.5, type=float)
    args = parser.parse_args()
    train(args)
