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

#### 2. Download Large Dataset  (If you want to train a language model to finish this task)

```bash
cd large_dataset
sudo apt install unzip
```

```bash
kaggle datasets download -d lizhecheng/llm-detect-ai-generated-text-dataset
unzip llm-detect-ai-generated-text-dataset.zip
```

#### 3. Download Traditional Dataset (If you want to use tree models and prompt-related dataset)
```bash
kaggle datasets download -d thedrcat/daigt-v2-train-dataset
unzip daigt-v2-train-dataset.zip
```

```bash
kaggle datasets download -d thedrcat/daigt-v3-train-dataset
unzip daigt-v3-train-dataset.zip
```

```bash
kaggle competitions download -c llm-detect-ai-generated-text
unzip llm-detect-ai-generated-text.zip
```


### Generate New Datasets

#### 1. Using gpt-3.5-16k (according to source texts, instructions and human written essays)

```bash
cd generate_dataset
(run real_texts_based.ipynb)
```


### Run Code

#### 1. Run Deberta Model (original version)

```bash
cd models
python deberta_trainer.py
```

#### 2. Run .sh File (set your own parameters)

```bash
cd models
chmod +x ./run.sh
./run.sh
```

#### 3.Run with AWP (set your own parameters)

```bash
cd models
chmod +x ./run_awp.sh
./run_awp.sh
```
#### 4.Run with Classification Models with Tf-idf Features.

```bash
cd models
python *_cls.py
```

#### 5. Run with Classification Models with Features from [Writing Quality Competition](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality)

```bash
cd models_essay_features
python *_cls.py
```
