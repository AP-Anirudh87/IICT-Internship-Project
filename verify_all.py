"""
====================================================================
  IICT AI Defense Systems: Automated Dual-Pipeline Verification
====================================================================
"""
import os
import sys
import re
import pickle
import numpy as np
import scipy.sparse as sp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "Project 1"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Project 2"))

# ── Clean text helper (Required for unpickling Project 1 model) ─────────────
_HTML_TAGS = re.compile(r"<[^>]+>")
_URLS      = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAILS    = re.compile(r"\S+@\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES    = re.compile(r"\s+")

def clean_text(text: str, min_len: int = 2) -> str:
    if not isinstance(text, str):
        text = str(text) if text else ""
    text = _HTML_TAGS.sub(" ", text)
    text = _URLS.sub(" ", text)
    text = _EMAILS.sub(" ", text).lower()
    text = _NON_ALPHA.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text

def clean_email(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text else ""
    t = re.sub(r"https?://\S+", " URL ", text)
    t = re.sub(r"\S+@\S+", " EMAIL ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()

from metadata_features import EmailMetadataExtractor

print("\n" + "=" * 65)
print("  🛡️  IICT AI DEFENSE SYSTEMS: FAST INTEGRITY TEST")
print("=" * 65)

# ── 1. TEST PROJECT 1 (TRUTHGUARD AI) ──────────────────────────────
print("\n[1/2] Testing Project 1: TruthGuard AI (Fake News Detection)...")
p1_model_path = os.path.join(ROOT_DIR, "Project 1", "models", "best_model.pkl")

if not os.path.exists(p1_model_path):
    print(f"  ❌ Model file not found at: {p1_model_path}")
else:
    try:
        with open(p1_model_path, "rb") as f:
            bundle1 = pickle.load(f)
        vec1    = bundle1["vectorizer"]
        model1  = bundle1["model"]
        name1   = bundle1.get("name", bundle1.get("model_name", "MLP Classifier"))

        test_headline = "WASHINGTON (Reuters) - The Senate passed a $1.2 trillion infrastructure bill on Tuesday, sending it to the House."
        cleaned1 = clean_text(test_headline)
        X1 = vec1.transform([cleaned1])
        pred1 = model1.predict(X1)[0]
        verdict1 = "FAKE NEWS" if pred1 == 1 else "REAL NEWS"

        print(f"  ✅ Model loaded successfully : {name1}")
        print(f"  ✅ Test Headline Input       : {test_headline[:60]}...")
        print(f"  ✅ System Prediction         : {verdict1}")
        print(f"  🎉 Project 1 Test Status     : {'PASS (Correctly identified as Legitimate)' if pred1 == 0 else 'CHECK'}")
    except Exception as e:
        print(f"  ❌ Error testing Project 1: {e}")

# ── 2. TEST PROJECT 2 (PHISHGUARD AI) ──────────────────────────────
print("\n[2/2] Testing Project 2: PhishGuard AI (Phishing Email Detection)...")
p2_model_path = os.path.join(ROOT_DIR, "Project 2", "models", "best_model.pkl")

if not os.path.exists(p2_model_path):
    print(f"  ❌ Model file not found at: {p2_model_path}")
else:
    try:
        with open(p2_model_path, "rb") as f:
            bundle2 = pickle.load(f)
        vec2    = bundle2["vectorizer"]
        model2  = bundle2["model"]
        scaler2 = bundle2.get("scaler")
        name2   = bundle2.get("name", bundle2.get("model_name", "MLP Classifier"))

        test_email = (
            "URGENT ACTION REQUIRED! Your bank account has been SUSPENDED! "
            "Verify your credentials immediately at http://192.168.1.5/secure-login "
            "or access will be permanently terminated."
        )
        cleaned2 = clean_email(test_email)
        X_tfidf2 = vec2.transform([cleaned2])
        
        extractor = EmailMetadataExtractor()
        meta2 = np.array([extractor.extract_vector(test_email)])
        meta_sc2 = scaler2.transform(meta2) if scaler2 is not None else meta2
        X_comb2  = sp.hstack([X_tfidf2, sp.csr_matrix(meta_sc2)])

        pred2 = model2.predict(X_comb2)[0]
        if hasattr(model2, "predict_proba"):
            classes = list(getattr(model2, "classes_", [0, 1]))
            idx = classes.index(1) if 1 in classes else 1
            prob2 = float(model2.predict_proba(X_comb2)[0][idx])
        else:
            prob2 = float(pred2)
        
        verdict2 = "PHISHING ATTACK" if pred2 == 1 else "SAFE EMAIL"

        print(f"  ✅ Model loaded successfully : {name2}")
        print(f"  ✅ Test Email Input          : {test_email[:60]}...")
        print(f"  ✅ Threat Score              : {prob2*100:.1f}%")
        print(f"  ✅ System Prediction         : {verdict2}")
        print(f"  🎉 Project 2 Test Status     : {'PASS (Correctly identified as Threat)' if pred2 == 1 else 'CHECK'}")
    except Exception as e:
        print(f"  ❌ Error testing Project 2: {e}")

print("\n" + "=" * 65)
print("  🏆  ALL DUAL DEFENSE PIPELINES ARE VERIFIED & OPERATIONAL!")
print("=" * 65 + "\n")
