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

```bash
kaggle competitions download -d lizhecheng/daigt-datasets
unzip daigt-datasets.zip
```


### Generate New Datasets

#### 1. Using GPT-3.5-16k (according to source texts, instructions and human written essays)

```bash
cd generate_dataset
(run real_texts_based.ipynb)
```

#### 2. Only Modify Human Text

```bash
cd generate_dataset
(run change_style_0117.ipynb)
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



## [121st Solution] You Can Achieve PB 0.929 Only Use TF-IDF

### 1. Start
Thanks to ``Kaggle`` and ``THE LEARNING AGENCY LAB`` for hosting this meaningful competition. In addition, I would like to thank all the ``Kagglers`` who shared datasets and innovative ideas. Although it's another drop on the private leaderboard, fortunately, I managed to hold on to the silver medal.

### 2. Finding

- ``n_grams = (3, 5)`` worked best for me, I did not try ``n_grams`` larger than ``5``.
- ``min_df = 2`` can boost scores of ``SGD`` and ``MultinomialNB`` almost ``0.02``, but would reduce scores of ``CatBoost`` and ``LGBM`` almost ``0.01``.
- When I used ``min_df = 2``, I tried up to ``57k`` data without encountering an out-of-memory error. However, when I didn't use ``min_df = 2``, I could only train a maximum of ``45k``.
- For ``SGD`` and ``MultinomialNB``, I created a new dataset combined [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset), [DAIGT V4 Magic Generations](https://www.kaggle.com/datasets/thedrcat/daigt-v4-train-dataset?select=daigt_magic_generations.csv), [Gemini Pro LLM - DAIGT](https://www.kaggle.com/datasets/asalhi/gemini-pro-llm-daigt), I could achieve LB score ``0.960`` with only these two models.
- For ``CatBoost`` and ``LGBM``, I still used original [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset), which could give great results on LB.
- I tried ``RandomForest`` on [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset), which can achieve LB score ``0.930``. Also, I tried ``MLP`` on the same dataset, got LB score ``0.939``.
- Reduce ``CatBoost`` iterations and increase learning rate can achieve better score and decrease a lot of execution time.

### 3. Final Code

I divided all the models into two major categories to generate prediction results since these two categories of models used different datasets and parameters. 

| Combo 1                  | Weights 1        | Combo 2                           | Weights 2           | Final Weights    | LB        | PB        | Chosen             |
| ------------------------ | ---------------- | --------------------------------- | ------------------- | ---------------- | --------- | --------- | ------------------ |
| ``(MultinomialNB, SGD)`` | ``[0.5, 0.5]``   | ``(LGBM, RandomForest)``          | ``[0.5, 0.5]``      | ``[0.4, 0.6]``   | ``0.970`` | ``0.907`` | ``Yes``            |
| ``(MultinomialNB, SGD)`` | ``[0.10, 0.31]`` | ``(LGBM, CatBoost)``              | ``[0.28, 0.67]``    | ``[0.3, 0.7]``   | ``0.966`` | ``0.908`` | ``Yes``            |
| ``(MultinomialNB, SGD)`` | ``[0.5, 0.5]``   | ``(CatBoost, RandomForest)``      | ``[20.0, 8.0]``     | ``[0.20, 0.80]`` | ``0.969`` | ``0.929`` | ``After Deadline`` |
| ``(MultinomialNB, SGD)`` | ``[0.5, 0.5]``   | ``(CatBoost, RandomForest, MLP)`` | ``[4.0, 1.5, 0.3]`` | ``[0.20, 0.80]`` | ``0.970`` | ``0.928`` | ``After Deadline`` |


Notebook Links:

[LB 0.970 PB 0.928 MNB+SGD+CB+RF+MLP](https://www.kaggle.com/code/lizhecheng/lb-0-970-pb-0-928-mnb-sgd-cb-rf-mlp/notebook)

[LB 0.969 PB 0.929 MNB+SGD+RF+CB](https://www.kaggle.com/code/lizhecheng/lb-0-969-pb-0-929-mnb-sgd-rf-cb/notebook)

As a result, although ``CatBoost`` score on the LB is relatively low compared to other models, it proves its strong robustness. Therefore, we can discover that giving ``CatBoost`` a higher weight can lead to better performance on the PB.

### 4. Not Work

- Set ``max_df`` or ``max_features`` did not work for me.

- I tried to generate new dataset by ``gpt-3.5-turbo``, but could not get a good result on my dataset.

  ```
  ## Here are three prompts I used to generate dataset.
  
  model_input = "The following is a human-written article. Now, please rewrite this article in your writing style, also optimize sentence structures and correct grammatical errors. You can appropriately add or remove content associated with the article, but should keep the general meaning unchanged. Just return the modified article.\n" + "article: " + human_text
  ```
  
- Tried ``SelectKBest`` and ``chi2`` to reduce the dimension of vectorized sparse matrix, LB score dropped.

  ```
  k = int(num_features / 4)
  chi2_selector = SelectKBest(chi2, k=k)
  X_train_chi2_selected = chi2_selector.fit_transform(X_train, y_train)
  X_test_chi2_selected = chi2_selector.transform(X_test)
  ```
  
- Tried ``TruncatedSVD`` too. However, since the dimension of original sparse matrix is too large, I could only set the new dimension to a very low number, which caused the LB score dropped a lot. (Setting a large output dimension for reduction can still lead to out-of-memory error because ``TruncatedSVD`` is achieved through matrix multiplication, which means that the generated new matrix also occupies memory space).

  ```
  n_components = int(num_features / 4)
  svd = TruncatedSVD(n_components=n_components)
  X_train_svd = svd.fit_transform(X_train)
  X_test_svd = svd.transform(X_test)
  ```

- Tried to use features from [last competition](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality), such as the ratio of word that length greater than 5, 6, ..., 10; the ratio of sentence that length greater than 25, 50, 75; different aggregations of word features, sentence features and paragraph features.

### 5. Conclusion

The robustness of large language models is indeed stronger than tree models. Additionally, in this competition, there is a higher requirement for the quality of training data for large language models. I used the publicly available large datasets from the discussions, but I did not achieve very ideal results. Therefore, it is essential to have the machine rewrite human-written articles to increase the model's discrimination difficulty.

I gained a lot from this competition and look forward to applying what I've learned in the next one. Team Avengers will keep moving forward.

### 6. Full Work

GitHub: [Here](https://github.com/Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text)