import sys
sys.path.append('.')
from phishing_email_detection import extract_metadata, clean_email, _URL_RE, _IP_URL_RE, URGENT_WORDS, MONEY_WORDS, _DIGIT_RE, _HTML_RE
import pickle, numpy as np, scipy.sparse as sp

with open('models/best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
vec = bundle['vectorizer']
scaler = bundle['scaler']

text = '''URGENT: Your PayPal account has been SUSPENDED due to suspicious activity!!! You must verify your identity immediately to restore access. If you do not click the link below within 24 hours, your account will be permanently closed and funds frozen. CLICK HERE NOW: http://192.168.1.5/paypal-secure-login'''

cleaned = clean_email(text)
X_tfidf = vec.transform([cleaned])
meta = np.array([extract_metadata(text)])
meta_sc = scaler.transform(meta)
X_combined = sp.hstack([X_tfidf, sp.csr_matrix(meta_sc)])

print(model.classes_)
print('Prob:', model.predict_proba(X_combined)[0])
print('Pred:', model.predict(X_combined)[0])
