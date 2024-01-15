import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
import optuna
import json
import sys
sys.path.append("..")
from essay_features_extractor import EssayProcessor
warnings.filterwarnings("ignore")

load_from_disk = False

print("... Loading Dataset ...")
df = pd.read_csv("../train_v3_drcat_02.csv", sep=",")
df = df[["text", "label", "prompt_name"]]
df["id"] = np.arange(len(df))
df = df.sample(1000)
print(df.head())

if not load_from_disk:
    print("... Constructing Features ...")
    essay_processor = EssayProcessor()
    train_sent_agg_df = essay_processor.sentence_processor(df=df)
    train_paragraph_agg_df = essay_processor.paragraph_processor(df=df)
    train_word_agg_df = essay_processor.word_processor(df=df)

    df_feats = train_sent_agg_df.merge(
        train_paragraph_agg_df, on="id", how="left"
    )
    df_feats = df_feats.merge(
        train_word_agg_df, on="id", how="left"
    )
    df_feats = df_feats.merge(df, on="id", how="left")
    # df_feats = df_feats.fillna(0.0)
    df_feats.reset_index(drop=True, inplace=True)
    print("The shape after constructing features:", df_feats.shape)

    print("... Saving Dataset ...")
    df_feats.to_csv("feats.csv", index=False)

else:
    print("... Loading Features From Disk ...")
    df_feats = pd.read_csv("feats.csv")

target_col = ["label"]
drop_cols = ["id", "prompt_name", "text", "sent", "paragraph", "word"]
train_cols = [
    col for col in df_feats.columns if col not in target_col + drop_cols
]

X_train = df_feats[train_cols]
y_train = df_feats[target_col]

print("... Optuna ...")


def objective(trial):
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_params = {
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
        "num_leaves": trial.suggest_int("num_leaves", 5, 50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
        "n_jobs": 4,
        "n_estimators": trial.suggest_int("n_estimators", 1000, 20000)
    }

    print("Params in this trial: ", best_params)

    val_fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**best_params)
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

        val_fold_score = roc_auc_score(y_val_fold, y_val_fold_pred)
        val_fold_scores.append(val_fold_score)
        print(f"Fold {fold + 1}, Fold Validation AUC: {val_fold_score}")

    average_val_fold_score = np.mean(val_fold_scores)
    print(f"Average Fold Validation AUC: {average_val_fold_score}")

    return average_val_fold_score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

with open("./lgbm_best_params.json", "w") as json_file:
    json.dump(trial.params, json_file, indent=4)

print("Save LightGBM best_params to json file")
