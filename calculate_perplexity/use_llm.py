from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

train = pd.read_csv("../train_v3_drcat_02.csv", sep=",")
train = train.drop_duplicates(subset=["text"])
train.reset_index(drop=True, inplace=True)
print("The shape of total dataset:", train.shape)


def calculate_perplexity(text):
    tokens = tokenizer.encode(
        text, return_tensors='pt',
        max_length=1024, padding="max_length"
    )
    output = model(tokens, labels=tokens)
    loss = output.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


df = pd.DataFrame(
    columns=[
        "text", "label", "prompt_name", "perplexity"
    ]
)


for idx, row in tqdm(train.iterrows(), total=len(train)):
    text = row["text"]

    new_row = pd.DataFrame({
        "text": [text],
        "label": [row["label"]],
        "prompt_name": [row["prompt_name"]],
        "perplexity": [calculate_perplexity(text)]
    })
    df = pd.concat([df, new_row], ignore_index=True)

df.reset_index(drop=True, inplace=True)
df.to_csv("llm_perplexity_results.csv", index=False)
