"""
========================================================
  Fake News Detection – Full Training Pipeline
  IICT Summer Internship Project 1
========================================================
Runs the complete 4-week pipeline headlessly:
  • Loads and preprocesses the dataset
  • Extracts TF-IDF features (sklearn)
  • Trains KNN, Logistic Regression, Random Forest, MLP
  • Evaluates and prints a comparison table
  • Saves plots to plots/
  • Saves the best model to models/best_model.pkl

Usage:
  python train_all.py
"""

import os
import sys
import pickle
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from text_preprocessor import TextPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble    import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ── Dark plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#0f3460", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.titlecolor": "#e94560",
    "axes.grid": True, "grid.color": "#0f3460", "grid.alpha": 0.4,
    "font.size": 11,
})
ACCENT = "#e94560"; BLUE = "#0f3460"; TEAL = "#53d8fb"; GOLD = "#f5a623"

os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 1 – Loading Dataset")
print("="*65)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
if not os.path.exists(DATA_PATH):
    print("[ERROR] Dataset not found. Run download_data.py first.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"  Shape   : {df.shape}")
print(f"  Columns : {df.columns.tolist()}")

# Normalise columns – support both schemas
#   Kaggle: id / title / author / text / label  (0=Real, 1=Fake)
#   lutzhamel: Unnamed:0 / title / text / label (REAL/FAKE string)
if "label" in df.columns:
    df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().upper() in ["FAKE", "1"] else 0)

for col in ["title", "text"]:
    if col in df.columns:
        df[col] = df[col].fillna("")
    else:
        df[col] = ""

if "author" not in df.columns:
    df["author"] = ""
else:
    df["author"] = df["author"].fillna("")

df.dropna(subset=["label"], inplace=True)
print(f"  Labels  : {df['label'].value_counts().to_dict()}  (0=Real, 1=Fake)")

# ── EDA plots ─────────────────────────────────────────────────────────────────
print("\n[INFO] Generating EDA plots …")
counts = df["label"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Class Distribution – Fake vs. Real News", fontsize=14, color=ACCENT)
bars = axes[0].bar(["Real (0)","Fake (1)"], counts.values, color=[TEAL, ACCENT], edgecolor="white", lw=0.6)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 f"{val:,}", ha="center", va="bottom", fontsize=11)
axes[0].set_ylabel("Count")
axes[1].pie(counts.values, labels=["Real","Fake"], colors=[TEAL, ACCENT],
            autopct="%1.1f%%", startangle=90, textprops={"color":"white"})
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/class_distribution.png")

# Text length distribution
df["text_len"] = df["text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(12, 5))
for lval, color, name in [(0, TEAL, "Real"), (1, ACCENT, "Fake")]:
    ax.hist(df[df["label"]==lval]["text_len"].clip(0, 2000), bins=60,
            alpha=0.7, color=color, label=name)
ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
ax.set_title("Text Length Distribution by Class"); ax.legend()
plt.tight_layout()
plt.savefig("plots/text_length_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/text_length_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 2 – Text Preprocessing & Feature Extraction")
print("="*65)

tp = TextPreprocessor()
df["combined"] = df["title"] + " " + df["author"] + " " + df["text"]
print("  Cleaning text (may take a moment) …")
df["cleaned"] = [tp.process(t) for t in df["combined"]]

X = df["cleaned"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"  TF-IDF matrix: {X_train_tfidf.shape}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 3 – Training Models")
print("="*65)

def train_eval(name, model, X_tr, X_te, y_tr, y_te):
    print(f"\n  [{name}] training …", end=" ", flush=True)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
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

m, p, met = train_eval("KNN (k=5)",
    KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
    X_train_tfidf, X_test_tfidf, y_train, y_test)
models["KNN"] = m; preds["KNN"] = p; results.append(met)

m, p, met = train_eval("Logistic Regression",
    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1),
    X_train_tfidf, X_test_tfidf, y_train, y_test)
models["LR"] = m; preds["LR"] = p; results.append(met)

m, p, met = train_eval("Random Forest",
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    X_train_tfidf, X_test_tfidf, y_train, y_test)
models["RF"] = m; preds["RF"] = p; results.append(met)

m, p, met = train_eval("MLP Neural Network",
    MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                  solver="adam", max_iter=30, random_state=42,
                  early_stopping=True, validation_fraction=0.1),
    X_train_tfidf, X_test_tfidf, y_train, y_test)
models["MLP"] = m; preds["MLP"] = p; results.append(met)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 4 – Evaluation & Plots")
print("="*65)

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n  ──── Model Comparison ────")
print(results_df.to_string())

# Bar chart
model_labels = list(results_df.index)
colors_ = [TEAL, ACCENT, GOLD, "#a78bfa"]
metrics_cols = ["Accuracy","Precision","Recall","F1-Score"]
x = np.arange(len(metrics_cols)); w = 0.18

fig, ax = plt.subplots(figsize=(13,6))
ax.set_title("Model Performance Comparison – Fake News Detection", fontsize=13)
for i, (name, color) in enumerate(zip(model_labels, colors_)):
    vals = results_df.loc[name, metrics_cols].values
    bars = ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + w*1.5); ax.set_xticklabels(metrics_cols)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.legend()
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/model_comparison.png")

# Confusion matrices
model_key_map = {"KNN (k=5)":"KNN","Logistic Regression":"LR",
                 "Random Forest":"RF","MLP Neural Network":"MLP"}
fig, axes = plt.subplots(2, 2, figsize=(14,11))
fig.suptitle("Confusion Matrices – All Models", fontsize=14, color=ACCENT)
for ax, name in zip(axes.flatten(), model_labels):
    key = model_key_map[name]
    cm  = confusion_matrix(y_test, preds[key])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", linewidths=0.5,
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(9,7))
ax.set_title("ROC Curves – All Models", fontsize=13)
for name, color in zip(model_labels, colors_):
    key = model_key_map[name]
    mod = models[key]
    if hasattr(mod, "predict_proba"):
        proba = mod.predict_proba(X_test_tfidf)[:,1]
    else:
        proba = preds[key].astype(float)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/roc_curves.png")

# Feature importance (Logistic Regression)
feature_names = tfidf.get_feature_names_out()
coefs = models["LR"].coef_[0]
top_n = 20
top_fake = coefs.argsort()[-top_n:][::-1]
top_real = coefs.argsort()[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(16,7))
fig.suptitle("Logistic Regression – Most Informative Features", fontsize=13, color=ACCENT)
axes[0].barh([feature_names[i] for i in reversed(top_real)],
              [abs(coefs[i]) for i in reversed(top_real)], color=TEAL)
axes[0].set_title("Top 20 → Real News"); axes[0].set_xlabel("|Coefficient|")
axes[1].barh([feature_names[i] for i in reversed(top_fake)],
              [coefs[i] for i in reversed(top_fake)], color=ACCENT)
axes[1].set_title("Top 20 → Fake News"); axes[1].set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/feature_importance.png")

# ── Save best model ────────────────────────────────────────────────────────────
best_name = results_df["F1-Score"].idxmax()
best_key  = model_key_map[best_name]
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": tfidf,
        "model"     : models[best_key],
        "name"      : best_name,
    }, f)

print(f"\n  ★  Best model  : {best_name}")
print(f"  ★  F1-Score    : {results_df.loc[best_name,'F1-Score']:.4f}")
print(f"  ★  Saved to    : models/best_model.pkl")

# Write results CSV
results_df.to_csv("models/results_summary.csv")
print("  ★  Results CSV : models/results_summary.csv")

print("\n" + "="*65)
print("  🏁  Project 1 training complete!")
print("="*65 + "\n")
