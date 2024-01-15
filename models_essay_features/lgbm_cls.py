import sys
sys.path.append("..")
import warnings
import pandas as pd
import numpy as np
from essay_features_extractor import EssayProcessor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
warnings.filterwarnings("ignore")

load_from_disk = False

print("... Loading Dataset ...")
df = pd.read_csv("../train_v3_drcat_02.csv")
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

print("... Training ...")

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

params = {
    "n_estimators": 10000,
    "verbose": 1,
    "objective": "cross_entropy",
    "metric": "auc",
    "learning_rate": 0.005,
    "colsample_bytree": 0.8,
    "random_state": 42,
    # "colsample_bynode": 0.8,
    # "lambda_l1": 4.562963348932286,
    # "lambda_l2": 2.97485,
    # "min_data_in_leaf": 115,
    # "max_depth": 23,
    # "max_bin": 898
}

val_fold_scores = []
# val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

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
    # val_score = roc_auc_score(y_val, y_val_pred)
    val_fold_scores.append(val_fold_score)
    # val_scores.append(val_score)
    print(
        f"Fold {fold + 1}, Fold Validation AUC: {val_fold_score}")

average_val_fold_score = np.mean(val_fold_scores)
# average_val_score = np.mean(val_scores)
print(
    f"Average Fold Validation AUC: {average_val_fold_score}")

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
# y_val_pred = model.predict_proba(X_val)[:, -1]
# final_val_score = roc_auc_score(y_val, y_val_pred)
# print(f"Final Validation Accuracy: {final_val_score}")

feature_importances = model.feature_importances_
feature_names = X_train.columns
features = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})
top_features = features.sort_values(by="Importance", ascending=False).head(50)
print(top_features)
