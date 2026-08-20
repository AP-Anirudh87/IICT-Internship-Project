"""
========================================================
  Phishing Email Detection – Full Training Pipeline
  IICT Summer Internship Project 2
========================================================
Runs the complete pipeline headlessly:
  • Loads and preprocesses the dataset
  • Extracts TF-IDF + metadata features
  • Trains NB, LR, RF, MLP
  • Evaluates and prints results table
  • Saves all plots to plots/
  • Saves the best model to models/best_model.pkl

Usage:
  python train_all.py
"""

import os
import sys
import re
import pickle
import warnings
import numpy  as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from metadata_features import extract_metadata_matrix, EmailMetadataExtractor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes   import ComplementNB
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ── Dark plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d1a", "axes.facecolor": "#111127",
    "axes.edgecolor": "#1a1a3e", "text.color": "#e8e8ff",
    "axes.labelcolor": "#e8e8ff", "xtick.color": "#e8e8ff",
    "ytick.color": "#e8e8ff", "axes.titlecolor": "#00d4ff",
    "axes.grid": True, "grid.color": "#1a1a3e", "grid.alpha": 0.4,
    "font.size": 11,
})
CYAN="#00d4ff"; ORANGE="#ff6b35"; GREEN="#39ff14"; PURPLE="#b44be1"

os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 1 – Loading Dataset")
print("="*65)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Phishing_Email.csv")
if not os.path.exists(DATA_PATH):
    print("[ERROR] Dataset not found. Run download_data.py first.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df.rename(columns={"Email Text": "text", "Email Type": "label"}, inplace=True)
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
df.dropna(subset=["text"], inplace=True)
df["text"]      = df["text"].astype(str)
df["label_int"] = (df["label"] == "Phishing Email").astype(int)
print(f"  Rows   : {len(df):,}")
print(f"  Labels : {df['label'].value_counts().to_dict()}")

# ── EDA plots ─────────────────────────────────────────────────────────────────
print("\n[INFO] Generating EDA plots …")
counts = df["label"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Class Distribution – Phishing vs. Safe Email", fontsize=14, color=CYAN)
axes[0].bar(counts.index, counts.values, color=[GREEN, ORANGE], edgecolor="white", lw=0.6)
axes[0].set_title("Email Count by Class")
axes[1].pie(counts.values, labels=counts.index, colors=[GREEN, ORANGE],
            autopct="%1.1f%%", startangle=90, textprops={"color":"white"})
axes[1].set_title("Class Balance")
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/class_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 2 – Preprocessing & Feature Engineering")
print("="*65)

def clean_email(t: str) -> str:
    t = re.sub(r"https?://\S+", " URL ", t)
    t = re.sub(r"\S+@\S+", " EMAIL ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()

df["cleaned"] = df["text"].apply(clean_email)

X_text = df["cleaned"].values
X_raw  = df["text"].values
y      = df["label_int"].values

(X_text_train, X_text_test,
 X_raw_train,  X_raw_test,
 y_train,      y_test) = train_test_split(
    X_text, X_raw, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train : {len(X_text_train):,}  |  Test : {len(X_text_test):,}")

print("  Building TF-IDF matrix …")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_text_train)
X_test_tfidf  = tfidf.transform(X_text_test)
print(f"  TF-IDF shape : {X_train_tfidf.shape}")

print("  Extracting metadata features …")
M_train = extract_metadata_matrix(list(X_raw_train))
M_test  = extract_metadata_matrix(list(X_raw_test))
scaler  = MinMaxScaler()
M_train_sc = scaler.fit_transform(M_train)
M_test_sc  = scaler.transform(M_test)

X_train_combined = sp.hstack([X_train_tfidf, sp.csr_matrix(M_train_sc)])
X_test_combined  = sp.hstack([X_test_tfidf,  sp.csr_matrix(M_test_sc)])
print(f"  Combined shape : {X_train_combined.shape}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 3 – Training Models")
print("="*65)

def train_eval(name, model, X_tr, X_te, y_tr, y_te, dense=False):
    print(f"\n  [{name}] training …", end=" ", flush=True)
    Xtr = X_tr.toarray() if dense and sp.issparse(X_tr) else X_tr
    Xte = X_te.toarray() if dense and sp.issparse(X_te) else X_te
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)
    met = {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall"   : recall_score(y_te, y_pred, zero_division=0),
        "F1-Score" : f1_score(y_te, y_pred, zero_division=0),
    }
    print(f"  Accuracy={met['Accuracy']:.4f}  F1={met['F1-Score']:.4f}")
    return model, y_pred, met

results = []; models = {}; preds = {}

m, p, met = train_eval("Naive Bayes",
    ComplementNB(alpha=0.1), X_train_tfidf, X_test_tfidf, y_train, y_test)
models["NB"] = m; preds["NB"] = p; results.append(met)

m, p, met = train_eval("Logistic Regression",
    LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1),
    X_train_combined, X_test_combined, y_train, y_test)
models["LR"] = m; preds["LR"] = p; results.append(met)

m, p, met = train_eval("Random Forest",
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    X_train_combined, X_test_combined, y_train, y_test)
models["RF"] = m; preds["RF"] = p; results.append(met)

m, p, met = train_eval("MLP Neural Network",
    MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                  solver="adam", max_iter=30, random_state=42,
                  early_stopping=True, validation_fraction=0.1),
    X_train_combined, X_test_combined, y_train, y_test, dense=True)
models["MLP"] = m; preds["MLP"] = p; results.append(met)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 4 – Evaluation & Plots")
print("="*65)

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n  ──── Model Comparison ────")
print(results_df.to_string())

model_labels = list(results_df.index)
colors_ = [CYAN, ORANGE, GREEN, PURPLE]
metrics_cols = ["Accuracy","Precision","Recall","F1-Score"]
x = np.arange(len(metrics_cols)); w = 0.18
model_key_map = {"Naive Bayes":"NB","Logistic Regression":"LR",
                 "Random Forest":"RF","MLP Neural Network":"MLP"}

# Bar chart
fig, ax = plt.subplots(figsize=(14,6))
ax.set_title("Phishing Email Detection – Model Comparison", fontsize=13, color=CYAN)
for i, (name, color) in enumerate(zip(model_labels, colors_)):
    vals = results_df.loc[name, metrics_cols].values
    bars = ax.bar(x+i*w, vals, w, label=name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x+w*1.5); ax.set_xticklabels(metrics_cols)
ax.set_ylim(0,1.12); ax.set_ylabel("Score"); ax.legend()
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/model_comparison.png")

# Confusion matrices
fig, axes = plt.subplots(2,2, figsize=(14,11))
fig.suptitle("Confusion Matrices – All Models", fontsize=14, color=CYAN)
for ax, name in zip(axes.flatten(), model_labels):
    key = model_key_map[name]
    cm  = confusion_matrix(y_test, preds[key])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", linewidths=0.5,
                xticklabels=["Safe","Phishing"], yticklabels=["Safe","Phishing"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(9,7))
ax.set_title("ROC Curves – Phishing Detection", fontsize=13, color=CYAN)
for name, color in zip(model_labels, colors_):
    key = model_key_map[name]
    Xte = X_test_tfidf if key == "NB" else X_test_combined
    if hasattr(models[key], "predict_proba"):
        if sp.issparse(Xte) and key == "MLP":
            proba = models[key].predict_proba(Xte.toarray())[:,1]
        else:
            proba = models[key].predict_proba(Xte)[:,1]
    else:
        proba = preds[key].astype(float)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC={auc(fpr,tpr):.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/roc_curves.png")

# Metadata feature importance (Random Forest)
meta_names = EmailMetadataExtractor.FEATURE_NAMES
all_feat   = list(tfidf.get_feature_names_out()) + meta_names
importances = models["RF"].feature_importances_
top_idx     = importances.argsort()[-20:][::-1]
top_names   = [all_feat[i] if i < len(all_feat) else f"feat_{i}" for i in top_idx]

fig, ax = plt.subplots(figsize=(11,7))
ax.barh(list(reversed(top_names)), list(reversed(importances[top_idx])), color=CYAN)
ax.set_title("Random Forest – Top 20 Feature Importances", color=CYAN, fontsize=13)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/feature_importance.png")

# Metadata distributions (sample)
sample_df = df.sample(min(3000, len(df)), random_state=42).copy()
ext = EmailMetadataExtractor()
for feat in ext.FEATURE_NAMES:
    sample_df[feat] = sample_df["text"].apply(lambda t: ext.extract(t)[feat])

fig, axes = plt.subplots(1,3, figsize=(16,5))
fig.suptitle("Metadata Feature Distributions", fontsize=13, color=CYAN)
for ax, feat, title in [
    (axes[0],"exclamation_count","Exclamation Marks"),
    (axes[1],"urgent_word_count","Urgency Words"),
    (axes[2],"uppercase_ratio",  "Uppercase Ratio"),
]:
    for lv, c, n in [("Safe Email",GREEN,"Safe"),("Phishing Email",ORANGE,"Phishing")]:
        ax.hist(sample_df[sample_df["label"]==lv][feat].clip(0,20), bins=20,
                alpha=0.7, color=c, label=n)
    ax.set_title(title); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/metadata_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/metadata_distributions.png")

# ── Save best model ────────────────────────────────────────────────────────────
best_name = results_df["F1-Score"].idxmax()
best_key  = model_key_map[best_name]
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": tfidf,
        "scaler"    : scaler,
        "model"     : models[best_key],
        "name"      : best_name,
    }, f)

print(f"\n  ★  Best model  : {best_name}")
print(f"  ★  F1-Score    : {results_df.loc[best_name,'F1-Score']:.4f}")
print(f"  ★  Saved to    : models/best_model.pkl")

results_df.to_csv("models/results_summary.csv")
print("  ★  Results CSV : models/results_summary.csv")

print("\n" + "="*65)
print("  🏁  Project 2 training complete!")
print("="*65 + "\n")
