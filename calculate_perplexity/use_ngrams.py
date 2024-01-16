import sys
import gc
import pandas as pd
import numpy as np
import math
import warnings
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
from collections import Counter
warnings.filterwarnings("ignore")

print("... Loading Dataset ...")
train = pd.read_csv("../combined_dataset.csv", sep=",")
excluded_prompt_name_list = [
    "Phones and driving", "Summer projects", "Mandatory extracurricular activities", "Community service",
    "Grades for extracurricular activities", "Cell phones at school", "Distance learning", "Seeking multiple opinions"
]
train = train[~(train["prompt_name"].isin(excluded_prompt_name_list))]
train = train.drop_duplicates(subset=["text"])
train.reset_index(drop=True, inplace=True)
print("The shape of total dataset:", train.shape)

human = train[train["label"] == 0]
human.reset_index(drop=True, inplace=True)
print("The shape of human dataset:", human.shape)

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
dataset = Dataset.from_pandas(human[["text"]])


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

tokenized_texts_train = []

for text in tqdm(train["text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


def build_ngrams(tokens, n):
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams


def calculate_perplexity(text, n):
    tokens = tokenizer.tokenize(text)

    ngrams = build_ngrams(tokens, n)
    n_minus_one_grams = build_ngrams(tokens, n - 1)

    ngram_freq = Counter(ngrams)
    n_minus_one_freq = Counter(n_minus_one_grams)

    perplexity = 1.0
    for ngram in ngrams:
        prefix = ngram[:-1]
        conditional_prob = ngram_freq[ngram] / \
            n_minus_one_freq[prefix] if n_minus_one_freq[prefix] > 0 else 0
        perplexity *= 1 / conditional_prob if conditional_prob > 0 else 0

    perplexity = perplexity ** (1 / len(ngrams)) \
        if len(ngrams) > 0 else float("inf")
    return perplexity


df = pd.DataFrame(
    columns=[
        "text", "label", "prompt_name", "2-grams per",
        "3-grams per", "4-grams per", "5-grams per"
    ]
)

for idx, row in tqdm(train.iterrows(), total=len(train)):
    text = row["text"]
    pp2 = calculate_perplexity(text, 2)
    pp3 = calculate_perplexity(text, 3)
    pp4 = calculate_perplexity(text, 4)
    pp5 = calculate_perplexity(text, 5)

    new_row = pd.DataFrame({
        "text": [text],
        "label": [row["label"]],
        "prompt_name": [row["prompt_name"]],
        "2-grams per": [pp2],
        "3-grams per": [pp3],
        "4-grams per": [pp4],
        "5-grams per": [pp5]
    })
    df = pd.concat([df, new_row], ignore_index=True)

df.reset_index(drop=True, inplace=True)
df.to_csv("perplexity_results.csv", index=False)
