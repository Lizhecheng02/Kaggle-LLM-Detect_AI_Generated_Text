## This Repo is for [Kaggle - LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)



### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```



### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Download Dataset

```bash
cd large_dataset
sudo apt install unzip
kaggle datasets download -d starblasters8/human-vs-llm-text-corpus
unzip human-vs-llm-text-corpus.zip
cd ..
```



### Run Code

#### 1.  Run Deberta Model

```bash
cd models
python deberta_trainer.py
```
