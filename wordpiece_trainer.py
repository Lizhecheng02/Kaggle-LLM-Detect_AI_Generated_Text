import pandas as pd
import gc
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from tqdm import tqdm
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import numpy as np


class CFG:
    is_train_on_full = False
    half_train_sample = 10000
    random_state = 42
    LOWER_CASE = False
    VOCAB_SIZE = 30522


train = pd.read_csv("train_v3_drcat_02.csv", sep=",")
test = pd.read_csv("test_essays.csv")

if CFG.is_train_on_full:
    print("-----Using full training data-----")
    train = train.drop_duplicates(subset=["text"])
    train = train.sample(len(train))
    print("The shape of training dataset is:", train.shape)
    train.reset_index(drop=True, inplace=True)
    print(train.head())
else:
    print("-----Using partial training data-----")
    train_label_0 = train[train["label"] == 0]
    train_label_1 = train[train["label"] == 1]
    train_label_0 = train_label_0.sample(
        CFG.half_train_sample, random_state=CFG.random_state
    )
    train_label_1 = train_label_1.sample(
        CFG.half_train_sample, random_state=CFG.random_state
    )
    train = pd.concat([train_label_0, train_label_1])
    train = train.sample(len(train))
    print("The shape of training dataset is:", train.shape)
    train.reset_index(drop=True, inplace=True)
    print(train.head())

raw_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if CFG.LOWER_CASE else []
)
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(
    vocab_size=CFG.VOCAB_SIZE,
    special_tokens=special_tokens
)

dataset = Dataset.from_pandas(test[["text"]])


def train_corpus():
    for i in tqdm(range(0, len(dataset), 100)):
        yield dataset[i:i + 100]["text"]


raw_tokenizer.train_from_iterator(train_corpus(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

save_directory = "./wordpiece_trained_tokenizer"
tokenizer.save_pretrained(save_directory)

tokenized_texts_test = []
for text in tqdm(test["text"].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []
for text in tqdm(train["text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


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
    strip_accents="unicode"
)

vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_
print(len(vocab))

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
y_train = train["label"].values
X_test = vectorizer.transform(tokenized_texts_test)

# 保存
save_npz("processed_data/wordpiece/X_train.npz", X_train)
save_npz("processed_data/wordpiece/X_test.npz", X_test)
np.save("processed_data/wordpiece/y_train.npy", y_train)

del vectorizer
gc.collect()
