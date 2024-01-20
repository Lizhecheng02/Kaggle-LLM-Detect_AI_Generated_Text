import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import optuna

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast


print("... Loading Dataset ...")
train = pd.read_csv("../kaggle_dataset/v2_gemini_magic.csv", sep=",")
# excluded_prompt_name_list = [
#     "Phones and driving", "Summer projects", "Mandatory extracurricular activities", "Community service",
#     "Grades for extracurricular activities", "Cell phones at school", "Distance learning", "Seeking multiple opinions"
# ]
# train = train[~(train["prompt_name"].isin(excluded_prompt_name_list))]
train = train.drop_duplicates(subset=["text"])
train.reset_index(drop=True, inplace=True)
print("The shape of total dataset:", train.shape)

print("... Splitting Dataset ...")

train_df, val_df = train_test_split(train, test_size=0.15, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print("The shape of training dataset:", train_df.shape)
print("The shape of validation dataset:", val_df.shape)

LOWERCASE = False
VOCAB_SIZE = 30522

print("... Training Tokenizer ...")
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
)
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE, special_tokens=special_tokens
)
dataset = Dataset.from_pandas(val_df[["text"]])


def train_corp_iter():
    for i in range(0, len(dataset), 100):
        yield dataset[i: i + 100]["text"]


raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_val = []

for text in tqdm(val_df["text"].tolist()):
    tokenized_texts_val.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in tqdm(train_df["text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

print("... Vectorizing ...")


def dummy(text):
    return text


vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
    min_df=2
)

vectorizer.fit(tokenized_texts_val)
vocab = vectorizer.vocabulary_
print("The number of vocabularies:", len(vocab))

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    vocabulary=vocab,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode"
)

X_train = vectorizer.fit_transform(tokenized_texts_train)
y_train = train_df["label"].values
X_val = vectorizer.transform(tokenized_texts_val)
y_val = val_df["label"].values
print("The shape of X_train is:", X_train.shape)
print("The shape of y_train is:", y_train.shape)
print("The shape of X_val is:", X_val.shape)
print("The shape of y_val is:", y_val.shape)

del vectorizer
gc.collect()

print("... Training ...")


def objective(trial):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "random_state": 42,
    }

    val_fold_scores = []
    val_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_train_fold, y_train_fold)

        y_val_fold_pred = model.predict_proba(X_val_fold)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        val_fold_score = roc_auc_score(y_val_fold, y_val_fold_pred)
        val_score = roc_auc_score(y_val, y_val_pred)

        val_fold_scores.append(val_fold_score)
        val_scores.append(val_score)

    average_val_fold_score = np.mean(val_fold_scores)
    average_val_score = np.mean(val_scores)

    return average_val_fold_score + average_val_score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best parameters:", best_params)
