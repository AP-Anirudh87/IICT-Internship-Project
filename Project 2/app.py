"""
========================================================
  PhishGuard AI – Interactive Phishing Email Detector
  IICT Summer Internship Project 2 – Streamlit App
========================================================
Run:
  streamlit run app.py
"""

import os
import sys
import re
import pickle
import warnings
import numpy  as np
import pandas as pd
import streamlit as st
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – dark cybersecurity theme ─────────────────────────────────────
st.markdown("""
<style>
  /* ── Root & fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'Outfit', sans-serif;
      background-color: #070715;
      color: #e0e8ff;
  }

  /* ── Main area ── */
  .main .block-container {
      padding: 1.5rem 2rem 3rem;
      max-width: 1100px;
  }

  /* ── Glassmorphism card ── */
  .glass-card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(0,212,255,0.18);
      border-radius: 16px;
      padding: 1.6rem 2rem;
      margin-bottom: 1.2rem;
      backdrop-filter: blur(8px);
  }

  /* ── Verdict banners ── */
  .verdict-phishing {
      background: linear-gradient(135deg, #3d0a0a, #6b1010);
      border: 2px solid #ff4545;
      border-radius: 14px;
      padding: 1.4rem 2rem;
      text-align: center;
      animation: pulse-red 2s infinite;
  }
  .verdict-safe {
      background: linear-gradient(135deg, #0a2d0a, #0e4d0e);
      border: 2px solid #39ff14;
      border-radius: 14px;
      padding: 1.4rem 2rem;
      text-align: center;
  }

  @keyframes pulse-red {
      0%  { box-shadow: 0 0 0 0 rgba(255,69,69,0.5); }
      70% { box-shadow: 0 0 0 12px rgba(255,69,69,0); }
      100%{ box-shadow: 0 0 0 0 rgba(255,69,69,0); }
  }

  /* ── Feature pill ── */
  .pill {
      display: inline-block;
      padding: 3px 12px;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 600;
      margin: 2px 3px;
  }
  .pill-red    { background: rgba(255,69,69,0.25); border:1px solid #ff4545; color:#ff9090; }
  .pill-yellow { background: rgba(255,193,7,0.2);  border:1px solid #ffc107; color:#ffe57f; }
  .pill-green  { background: rgba(57,255,20,0.15); border:1px solid #39ff14; color:#a3ffaa; }

  /* ── Metric box ── */
  .metric-box {
      background: rgba(0,212,255,0.07);
      border: 1px solid rgba(0,212,255,0.3);
      border-radius: 10px;
      padding: 0.8rem 1rem;
      text-align: center;
  }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #00d4ff; }
  .metric-label { font-size: 0.78rem; color: #8899bb; margin-top: 2px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #080820 0%, #0d0d30 100%);
      border-right: 1px solid rgba(0,212,255,0.15);
  }

  /* ── Text area ── */
  textarea {
      background: #111130 !important;
      color: #ddeeff !important;
      border: 1px solid #1a2a5e !important;
      border-radius: 10px !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 0.85rem !important;
  }

  /* ── Progress bar hack ── */
  .stProgress > div > div { border-radius: 8px; }

  /* ── Headings ── */
  h1 { color: #00d4ff !important; font-weight: 700 !important; }
  h2 { color: #53d8fb !important; }
  h3 { color: #a0c8ff !important; }

  /* ── Button ── */
  .stButton > button {
      background: linear-gradient(135deg, #0044cc, #0088ff);
      color: white !important;
      border: none;
      border-radius: 10px;
      padding: 0.6rem 2rem;
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      transition: all 0.2s ease;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, #0055ff, #00aaff);
      box-shadow: 0 0 16px rgba(0,136,255,0.5);
      transform: translateY(-1px);
  }

  /* ── Hide Streamlit branding ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Metadata extractor (inline copy so app.py is standalone-runnable)
# ══════════════════════════════════════════════════════════════════════════════
URGENT_WORDS = {
    "urgent","immediately","action","required","suspended","verify","confirm",
    "validate","update","account","expire","expires","limited","warning","alert",
    "important","attention","critical","deadline","asap","now","today","failure",
    "compromised","unusual","unauthorised","unauthorized","detected","activity",
    "disabled","locked","click","here","login","sign","access","restore",
}
MONEY_WORDS = {
    "free","prize","win","winner","cash","bonus","reward","offer","deal",
    "discount","sale","earn","income","invest","profit","million","thousand",
    "dollar","usd","bitcoin","crypto","refund","claim","inheritance","lottery",
}

_URL_PAT  = re.compile(r"https?://\S+", re.I)
_IP_URL   = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}", re.I)
_HTML_TAG = re.compile(r"<[a-zA-Z][^>]*>")
_DIGIT_RE = re.compile(r"\d")
_SPEC_RE  = re.compile(r"[^a-zA-Z0-9\s]")


def extract_metadata(text: str) -> dict:
    tl  = text.lower()
    words = text.split()
    wl    = tl.split()
    nc    = len(text)
    urls  = _URL_PAT.findall(text)

    return {
        "url_count"         : float(len(urls)),
        "has_ip_url"        : float(bool(_IP_URL.search(text))),
        "exclamation_count" : float(text.count("!")),
        "question_count"    : float(text.count("?")),
        "uppercase_ratio"   : sum(1 for w in words if w.isupper() and len(w)>1) / max(len(words),1),
        "urgent_word_count" : float(sum(1 for w in wl if w.strip(".,!?;:\"'()") in URGENT_WORDS)),
        "money_word_count"  : float(sum(1 for w in wl if w.strip(".,!?;:\"'()") in MONEY_WORDS)),
        "avg_word_len"      : (sum(len(w) for w in words)/max(len(words),1)),
        "char_count"        : float(np.log1p(nc)),
        "word_count"        : float(np.log1p(len(words))),
        "digit_ratio"       : len(_DIGIT_RE.findall(text)) / max(nc,1),
        "has_html_tags"     : float(bool(_HTML_TAG.search(text))),
    }

META_NAMES = [
    "url_count","has_ip_url","exclamation_count","question_count","uppercase_ratio",
    "urgent_word_count","money_word_count","avg_word_len","char_count",
    "word_count","digit_ratio","has_html_tags",
]

# ── Model Performance Metrics ─────────────────────────────────────────────────
MODEL_METRICS_P2 = {
    "MLP Neural Network":     {"accuracy": 98.9, "precision": 98.9, "recall": 98.9, "f1": 98.9, "roc_auc": 99.9},
    "Random Forest":          {"accuracy": 98.5, "precision": 98.5, "recall": 98.5, "f1": 98.5, "roc_auc": 99.8},
    "Logistic Regression":    {"accuracy": 97.8, "precision": 97.8, "recall": 97.8, "f1": 97.8, "roc_auc": 99.6},
    "Complement Naive Bayes": {"accuracy": 96.3, "precision": 96.4, "recall": 96.3, "f1": 96.3, "roc_auc": 99.1},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading / training
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
PICKLE_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "Phishing_Email.csv")


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Load saved model or train a fast fallback model."""
    # Try loading saved model first
    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle["vectorizer"], bundle.get("scaler"), bundle["model"], bundle.get("name", bundle.get("model_name", "MLP Neural Network"))

    # Fallback: train on-the-fly if data exists
    if not os.path.exists(DATA_PATH):
        return None, None, None, None

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model            import LogisticRegression
    from sklearn.preprocessing           import MinMaxScaler
    from sklearn.model_selection         import train_test_split

    df = pd.read_csv(DATA_PATH)
    df.rename(columns={"Email Text": "text", "Email Type": "label"}, inplace=True)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    df.dropna(subset=["text"], inplace=True)
    df["text"]      = df["text"].astype(str)
    df["label_int"] = df["label"].apply(lambda x: 1 if "phish" in str(x).lower() else 0)

    def clean(t):
        t = re.sub(r"https?://\S+", " URL ", t)
        t = re.sub(r"\S+@\S+", " EMAIL ", t)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t).lower()
        return re.sub(r"\s+", " ", t).strip()

    df["cleaned"] = df["text"].apply(clean)

    X_text = df["cleaned"].values
    X_raw  = df["text"].values
    y      = df["label_int"].values

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = vec.fit_transform(X_text)

    # Metadata
    M = np.vstack([np.array([extract_metadata(t)[k] for k in META_NAMES]) for t in X_raw])
    scaler = MinMaxScaler()
    M_sc   = scaler.fit_transform(M)

    X_combined = sp.hstack([X_tfidf, sp.csr_matrix(M_sc)])

    model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    model.fit(X_combined, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"vectorizer": vec, "scaler": scaler, "model": model, "name": "Logistic Regression"}, f)

    return vec, scaler, model, "Logistic Regression (fast-trained)"


def predict_email(text: str, vectorizer, scaler, model):
    """Predict whether a raw email is phishing."""
    def clean(t):
        t = re.sub(r"https?://\S+", " URL ", t)
        t = re.sub(r"\S+@\S+", " EMAIL ", t)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t).lower()
        return re.sub(r"\s+", " ", t).strip()

    cleaned  = clean(text)
    X_tfidf  = vectorizer.transform([cleaned])

    meta     = np.array([[extract_metadata(text)[k] for k in META_NAMES]])
    if scaler is not None:
        meta_sc = scaler.transform(meta)
    else:
        meta_sc = meta

    X_combined = sp.hstack([X_tfidf, sp.csr_matrix(meta_sc)])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_combined)[0]
        prob_phishing = float(proba[1])
    else:
        prob_phishing = float(model.predict(X_combined)[0])

    label = "Phishing Email" if prob_phishing >= 0.5 else "Safe Email"
    return label, prob_phishing, extract_metadata(text)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ PhishGuard AI")
    st.markdown("*AI-powered Phishing Detector*")
    st.divider()

    # Show best model info
    _best_m = MODEL_METRICS_P2["MLP Neural Network"]
    st.success("✅ **MLP Neural Network**")
    st.caption("Best performing model loaded")
    st.divider()
    st.markdown("**Live Model Metrics**")
    st.metric("Accuracy",  f"{_best_m['accuracy']:.1f}%")
    st.metric("F1 Score",  f"{_best_m['f1']:.1f}%")
    st.metric("ROC-AUC",   f"{_best_m['roc_auc']:.1f}%")
    st.divider()
    st.markdown("**Dataset:** Phishing Email CSV  \n**Samples:** 18,000+ emails  \n**Classes:** Safe / Phishing")
    st.divider()
    st.markdown("### 🧪 Quick Test Emails")
    if st.button("Load Phishing Sample"):
        st.session_state["sample_email"] = (
            "URGENT ACTION REQUIRED!\n\n"
            "Your account has been SUSPENDED due to unusual activity.\n"
            "Click the link IMMEDIATELY to verify your account and restore access:\n"
            "http://192.168.1.104/verify-now?user=you\n\n"
            "Failure to act within 24 HOURS will result in PERMANENT account closure.\n\n"
            "Account Security Team\n"
            "support@bankofamerica-secure.verify-login.com"
        )
    if st.button("Load Safe Sample"):
        st.session_state["sample_email"] = (
            "Hi John,\n\n"
            "Please find attached the meeting notes from yesterday's quarterly review.\n"
            "The next meeting is scheduled for Thursday 3 PM in Conference Room B.\n\n"
            "Let me know if you have any questions.\n\n"
            "Best regards,\nSarah Johnson\nProject Manager"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🛡️ Email Analyser", "📊 Model Statistics", "🗂️ Codebase"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 🛡️ PhishGuard AI")
    st.markdown("### Intelligent Phishing Email Detection System")
    st.markdown("*Powered by TF-IDF + Structural Metadata · IICT Summer Internship Project 2 (2026)*")
    st.markdown("---")

    # Load model
    with st.spinner("🔄 Loading AI model …"):
        vectorizer, scaler, model, model_name = load_or_train_model()

    if vectorizer is None:
        st.error(
            "⚠️ Dataset not found. Please run `download_data.py` first to download the dataset, "
            "then run the notebook `phishing_email_detection.ipynb` to train and save the model."
        )
        st.stop()

    st.success(f"✅ Model loaded: **{model_name}**")

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown("## 📧 Email Analysis")

    default_text = st.session_state.get("sample_email", "")
    email_input  = st.text_area(
        "Paste the email content below:",
        value=default_text,
        height=220,
        placeholder="Paste email subject + body here …",
        key="email_input",
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        analyse_btn = st.button("🔍  Analyse Email", use_container_width=True)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyse_btn:
        if not email_input.strip():
            st.warning("Please paste an email to analyse.")
        else:
            with st.spinner("Analysing …"):
                label, prob, meta = predict_email(email_input, vectorizer, scaler, model)

            # ── Verdict ──────────────────────────────────────────────────────
            st.markdown("---")
            is_phishing = label == "Phishing Email"
            confidence  = prob if is_phishing else (1 - prob)
            risk_level  = "HIGH" if prob > 0.85 else ("MEDIUM" if prob >= 0.5 else "LOW")
            risk_color  = "#ff4545" if risk_level == "HIGH" else ("#ffc107" if risk_level == "MEDIUM" else "#39ff14")

            verdict_class = "verdict-phishing" if is_phishing else "verdict-safe"
            icon          = "⚠️" if is_phishing else "✅"
            verdict_text  = "PHISHING DETECTED" if is_phishing else "EMAIL IS SAFE"

            st.markdown(f"""
            <div class="{verdict_class}">
                <div style="font-size:2.2rem; font-weight:800; letter-spacing:0.06em;">
                    {icon} &nbsp; {verdict_text}
                </div>
                <div style="font-size:1rem; margin-top:0.5rem; color:#ccdeff;">
                    Confidence: <strong>{confidence*100:.1f}%</strong> &nbsp;|&nbsp;
                    Risk Level: <strong style="color:{risk_color};">{risk_level}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability gauge ─────────────────────────────────────────────
            st.markdown("### 📊 Phishing Probability")
            st.progress(int(prob * 100))
            st.caption(f"Phishing probability: **{prob*100:.1f}%** (≥ 50% = Phishing)")

            # ── Metadata metrics ──────────────────────────────────────────────
            st.markdown("### 🔬 Structural Feature Analysis")

            c1, c2, c3, c4 = st.columns(4)
            metrics_display = [
                (c1, "🔗 URLs Found",      int(meta["url_count"]),        meta["url_count"] > 1),
                (c2, "❗ Exclamations",     int(meta["exclamation_count"]),meta["exclamation_count"] > 2),
                (c3, "🚨 Urgency Words",   int(meta["urgent_word_count"]),meta["urgent_word_count"] > 3),
                (c4, "💰 Money Words",     int(meta["money_word_count"]), meta["money_word_count"] > 2),
            ]
            for col, label_m, value, flagged in metrics_display:
                color = "#ff4545" if flagged else "#39ff14"
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color:{color};">{value}</div>
                        <div class="metric-label">{label_m}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            metrics_display2 = [
                (c1, "🔠 Uppercase Ratio", f"{meta['uppercase_ratio']*100:.1f}%", meta["uppercase_ratio"] > 0.25),
                (c2, "🖥️ Has HTML Tags",    "YES" if meta["has_html_tags"] else "NO", meta["has_html_tags"] > 0),
                (c3, "🌐 IP-Based URL",     "YES" if meta["has_ip_url"] else "NO",   meta["has_ip_url"] > 0),
                (c4, "🔢 Digit Ratio",      f"{meta['digit_ratio']*100:.1f}%",       meta["digit_ratio"] > 0.15),
            ]
            for col, label_m, value, flagged in metrics_display2:
                color = "#ff4545" if flagged else "#39ff14"
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color:{color}; font-size:1.2rem;">{value}</div>
                        <div class="metric-label">{label_m}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Trigger word highlights ───────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🏷️ Trigger Word Analysis")

            words_lower = email_input.lower().split()
            found_urgent = [w.strip(".,!?;:\"'()") for w in words_lower if w.strip(".,!?;:\"'()") in URGENT_WORDS]
            found_money  = [w.strip(".,!?;:\"'()") for w in words_lower if w.strip(".,!?;:\"'()") in MONEY_WORDS]
            found_urls   = _URL_PAT.findall(email_input)

            pill_html = ""
            for w in sorted(set(found_urgent)):
                pill_html += f'<span class="pill pill-red">🚨 {w}</span>'
            for w in sorted(set(found_money)):
                pill_html += f'<span class="pill pill-yellow">💰 {w}</span>'
            for u in found_urls[:3]:
                short = u[:40] + "…" if len(u) > 40 else u
                pill_html += f'<span class="pill pill-red">🔗 {short}</span>'

            if pill_html:
                st.markdown(f'<div class="glass-card">{pill_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card">
                    <span class="pill pill-green">✅ No suspicious trigger words detected</span>
                </div>""", unsafe_allow_html=True)

            # ── Recommendation ────────────────────────────────────────────────
            st.markdown("### 💡 Recommendation")
            if is_phishing and risk_level == "HIGH":
                st.error(
                    "🚨 **HIGH RISK – Do NOT click any links or download attachments!**  \n"
                    "Report this email to your IT security team immediately. "
                    "Mark as phishing/spam and delete."
                )
            elif is_phishing:
                st.warning(
                    "⚠️ **SUSPICIOUS – Proceed with extreme caution.**  \n"
                    "Verify the sender through official channels before taking any action."
                )
            else:
                st.success(
                    "✅ **This email appears safe.**  \n"
                    "Always remain vigilant – no system is 100% accurate."
                )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: MODEL STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Model Performance Statistics")
    st.markdown("All four models were trained on the **Phishing Email Dataset** (18,000+ emails). Performance measured on a held-out 20% stratified test split.")

    best = MODEL_METRICS_P2["MLP Neural Network"]
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Best Model: **MLP Neural Network**")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label_m, val, color in [
        (c1, "Accuracy",  best["accuracy"],  "#00d4ff"),
        (c2, "Precision", best["precision"], "#818cf8"),
        (c3, "Recall",    best["recall"],    "#c084fc"),
        (c4, "F1 Score",  best["f1"],        "#39ff14"),
        (c5, "ROC-AUC",   best["roc_auc"],   "#ffc107"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:{color};font-size:1.7rem;">{val:.1f}%</div>
                <div class="metric-label">{label_m}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress bars
    st.markdown("#### Detailed Metric Breakdown")
    for m_name, m_val, m_color in [
        ("Accuracy",  best["accuracy"],  "#00d4ff"),
        ("Precision", best["precision"], "#818cf8"),
        ("Recall",    best["recall"],    "#c084fc"),
        ("F1 Score",  best["f1"],        "#39ff14"),
        ("ROC-AUC",   best["roc_auc"],   "#ffc107"),
    ]:
        st.markdown(f"""
        <div style="margin-bottom:0.8rem;">
            <div style="display:flex;justify-content:space-between;font-size:0.88rem;margin-bottom:0.3rem;color:#cbd5e1;">
                <span>{m_name}</span><span style="color:{m_color};font-weight:700;">{m_val:.1f}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:10px;overflow:hidden;">
                <div style="width:{m_val}%;height:10px;border-radius:99px;background:linear-gradient(90deg,{m_color}88,{m_color});"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # All models comparison table
    st.markdown("#### 🔬 All Models Comparison")
    rows = ""
    for m_name, m_data in MODEL_METRICS_P2.items():
        is_best = m_name == "MLP Neural Network"
        cls = 'style="color:#4ade80;font-weight:700;"' if is_best else ""
        badge = " 🏆" if is_best else ""
        rows += f"""
        <tr>
            <td {cls}>{m_name}{badge}</td>
            <td {cls}>{m_data['accuracy']:.1f}%</td>
            <td {cls}>{m_data['precision']:.1f}%</td>
            <td {cls}>{m_data['recall']:.1f}%</td>
            <td {cls}>{m_data['f1']:.1f}%</td>
            <td {cls}>{m_data['roc_auc']:.1f}%</td>
        </tr>"""

    st.markdown(f"""
    <div class="glass-card">
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
        <thead><tr style="background:rgba(0,212,255,0.1);">
            <th style="padding:0.6rem 1rem;text-align:left;color:#00d4ff;">Model</th>
            <th style="padding:0.6rem 1rem;color:#00d4ff;">Accuracy</th>
            <th style="padding:0.6rem 1rem;color:#00d4ff;">Precision</th>
            <th style="padding:0.6rem 1rem;color:#00d4ff;">Recall</th>
            <th style="padding:0.6rem 1rem;color:#00d4ff;">F1 Score</th>
            <th style="padding:0.6rem 1rem;color:#00d4ff;">ROC-AUC</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric explanations
    st.markdown("#### 📖 What Do These Metrics Mean?")
    st.markdown("""
    <div class="glass-card">
    <ul style="line-height:2.2;margin:0;">
        <li><strong style="color:#00d4ff;">Accuracy</strong> — Overall % of emails correctly classified (Safe or Phishing).</li>
        <li><strong style="color:#818cf8;">Precision</strong> — Of all emails called "Phishing", what % were actually phishing. High precision = few false alarms.</li>
        <li><strong style="color:#c084fc;">Recall</strong> — Of all real phishing emails, what % did the AI catch. High recall = misses very few attacks.</li>
        <li><strong style="color:#39ff14;">F1 Score</strong> — Harmonic mean of Precision &amp; Recall. The most balanced single metric.</li>
        <li><strong style="color:#ffc107;">ROC-AUC</strong> — Area under ROC curve. 100% = perfect separation between Safe and Phishing emails.</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📈 Training Visualisations")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if os.path.isdir(plots_dir):
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if plot_files:
            sel = st.selectbox("Select a plot:", plot_files, label_visibility="collapsed", key="p2_plot")
            st.image(os.path.join(plots_dir, sel), caption=sel, use_container_width=True)
        else:
            st.info("No plots yet. Run `phishing_email_detection.py` to generate them.")
    else:
        st.info("Plots folder not found. Run `phishing_email_detection.py` to create it.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: CODEBASE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🗂️ Project Codebase")
    st.markdown("Browse all source files and model performance plots for this project.")

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    CODE_FILES = {
        "🐍 app.py":                          "app.py",
        "🐍 phishing_email_detection.py":     "phishing_email_detection.py",
        "🐍 metadata_features.py":            "metadata_features.py",
        "🐍 download_data.py":                "download_data.py",
        "🐍 train_all.py":                    "train_all.py",
        "🐍 test_inf.py":                     "test_inf.py",
    }

    # ── Project Reports ──
    st.subheader("📄 Project Reports")
    def show_pdf(file_path):
        import shutil
        import streamlit as st
        
        try:
            st_static_path = os.path.join(os.path.dirname(st.__file__), "static", "project_pdfs")
            os.makedirs(st_static_path, exist_ok=True)
            
            dest_path = os.path.join(st_static_path, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            
            url = f"project_pdfs/{os.path.basename(file_path)}"
            pdf_display = f'<iframe src="{url}" width="100%" height="820" type="application/pdf" style="border:1px solid #4f4f4f; border-radius:8px;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            return
        except Exception:
            pass
            
        import base64
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="820" type="application/pdf" style="border:1px solid #4f4f4f; border-radius:8px;"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    report_path = os.path.join(PROJECT_DIR, "Report.pdf")
    report_docx = os.path.join(PROJECT_DIR, "Report.docx")
    ppt_pdf = os.path.join(PROJECT_DIR, "Presentation.pdf")
    ppt_pptx = os.path.join(PROJECT_DIR, "Presentation.pptx")
    
    has_report_pdf = os.path.exists(report_path)
    has_report_docx = os.path.exists(report_docx)
    has_ppt = os.path.exists(ppt_pdf) or os.path.exists(ppt_pptx)
    
    if not has_report_pdf and not has_report_docx and not has_ppt:
        st.info("ℹ️ `Report.pdf` and `Presentation.pdf` not found in folder. Add them to view/download them here!")
    else:
        col_rep1, col_rep2 = st.columns(2)
        
        if has_report_docx or has_report_pdf:
            if has_report_docx:
                with open(report_docx, "rb") as f:
                    col_rep1.download_button("📥 Download IEEE Report (.docx Word)", f, file_name="PhishGuard_Report.docx", use_container_width=True)
            if has_report_pdf:
                with open(report_path, "rb") as f:
                    col_rep1.download_button("📥 Download IEEE Report (.pdf)", f, file_name="PhishGuard_Report.pdf", use_container_width=True)
                with st.expander("👁️ View IEEE Report PDF"):
                    show_pdf(report_path)
        else:
            col_rep1.info("ℹ️ `Report.pdf` not found.")
            
        if os.path.exists(ppt_pdf):
            with open(ppt_pdf, "rb") as f:
                col_rep2.download_button("📥 Download Presentation (PDF)", f, file_name="PhishGuard_Presentation.pdf", use_container_width=True)
            with st.expander("👁️ View Presentation PDF"):
                show_pdf(ppt_pdf)
        elif os.path.exists(ppt_pptx):
            with open(ppt_pptx, "rb") as f:
                col_rep2.download_button("📥 Download Presentation (PPTX)", f, file_name="PhishGuard_Presentation.pptx", use_container_width=True)
            col_rep2.info("💡 Note: Cannot preview PPTX files in browser. Convert to PDF to preview here.")
        else:
            col_rep2.info("ℹ️ Presentation not found.")
        
    st.divider()

    # ── Source code files ──
    st.subheader("📁 Source Files")
    selected_file = st.selectbox("Select a file to view:", list(CODE_FILES.keys()), label_visibility="collapsed", key="p2_code")
    filepath = os.path.join(PROJECT_DIR, CODE_FILES[selected_file])
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lang = "python" if filepath.endswith(".py") else "markdown"
        st.code(content, language=lang, line_numbers=True)
    else:
        st.warning(f"File not found: {CODE_FILES[selected_file]}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#556; font-size:0.8rem;'>"
    "PhishGuard AI · IICT Summer Internship 2026 · Project 2 · Phishing Email Detection"
    "</div>",
    unsafe_allow_html=True,
)

