import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

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
train = pd.read_csv("../train_v3_drcat_02.csv", sep=",")
excluded_prompt_name_list = [
    "Phones and driving", "Summer projects", "Mandatory extracurricular activities", "Community service",
    "Grades for extracurricular activities", "Cell phones at school", "Distance learning", "Seeking multiple opinions"
]
train = train[~(train["prompt_name"].isin(excluded_prompt_name_list))]
train = train.drop_duplicates(subset=["text"])
train.reset_index(drop=True, inplace=True)
print("The shape of total dataset:", train.shape)

print("... Splitting Dataset ...")

train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)
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

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

params = {
    "n_estimators": 2000,
    "verbose": 1,
    "objective": "cross_entropy",
    "metric": "auc",
    "learning_rate": 0.005,
    "colsample_bytree": 0.4,
    "random_state": 42,
    # "colsample_bynode": 0.8,
    # "lambda_l1": 4.562963348932286,
    # "lambda_l2": 2.97485,
    # "min_data_in_leaf": 115,
    # "max_depth": 23,
    # "max_bin": 898
}

val_fold_scores = []
val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**params)
    early_stopping_callback = lgb.early_stopping(
        50, first_metric_only=True, verbose=True
    )
    model.fit(
        X=X_train_fold,
        y=y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[early_stopping_callback]
    )

    y_val_fold_pred = model.predict_proba(
        X_val_fold, num_iteration=model.best_iteration_
    )[:, 1]
    y_val_pred = model.predict_proba(
        X_val_fold, num_iteration=model.best_iteration_
    )[:, 1]

    val_fold_score = roc_auc_score(y_val_fold, y_val_fold_pred)
    val_score = roc_auc_score(y_val, y_val_pred)
    val_fold_scores.append(val_fold_score)
    val_scores.append(val_score)
    print(
        f"Fold {fold + 1}, Fold Validation AUC: {val_fold_score}, Real Validation AUC: {val_score}")

average_val_fold_score = np.mean(val_fold_scores)
average_val_score = np.mean(val_scores)
print(
    f"Average Fold Validation AUC: {average_val_fold_score}, Average Validation AUC: {average_val_score}")

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
y_val_pred = model.predict_proba(X_val)[:, -1]
final_val_score = roc_auc_score(y_val, y_val_pred)
print(f"Final Validation Accuracy: {final_val_score}")

feature_importances = model.feature_importances_
feature_names = X_train.columns
features = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})
top_features = features.sort_values(by="Importance", ascending=False).head(50)
print(top_features)
