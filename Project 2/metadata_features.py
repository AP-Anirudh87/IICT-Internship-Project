"""
========================================================
  Phishing Email Detection – Metadata Feature Extractor
  IICT Summer Internship Project 2
========================================================
Extracts structural / behavioural features from raw email
text without relying on any NLP library. These features
complement TF-IDF bag-of-words representations:

  1.  url_count           – number of http/https links
  2.  has_ip_url          – link contains raw IP address
  3.  exclamation_count   – number of "!" characters
  4.  uppercase_ratio     – fraction of UPPERCASE words
  5.  urgent_word_count   – count of urgency trigger words
  6.  money_word_count    – count of financial trigger words
  7.  avg_word_len        – average token length
  8.  char_count          – total character count
  9.  word_count          – total word count
  10. digit_ratio         – fraction of digit characters
  11. special_char_count  – non-alphanumeric, non-space chars
  12. has_html_tags       – presence of <a>, <img>, etc.
"""

import re
import math
import numpy as np
from typing import List, Dict, Union

# ── Trigger word lists ────────────────────────────────────────────────────────
URGENT_WORDS = {
    "urgent", "immediately", "action", "required", "suspended", "verify",
    "confirm", "validate", "update", "account", "expire", "expires",
    "limited", "warning", "alert", "important", "attention", "critical",
    "deadline", "asap", "now", "today", "failure", "compromised",
    "unusual", "unauthorised", "unauthorized", "detected", "activity",
    "suspended", "disabled", "locked", "click", "here", "login",
    "sign in", "access", "restore",
}

MONEY_WORDS = {
    "free", "prize", "win", "winner", "cash", "bonus", "reward",
    "offer", "deal", "discount", "sale", "earn", "income", "invest",
    "profit", "million", "thousand", "dollar", "usd", "bitcoin",
    "crypto", "refund", "claim", "inheritance", "lottery",
}

# ── Compiled regex patterns ────────────────────────────────────────────────────
_URL_PATTERN        = re.compile(r"https?://\S+", re.IGNORECASE)
_IP_IN_URL          = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}", re.IGNORECASE)
_HTML_TAG           = re.compile(r"<[a-zA-Z][^>]*>")
_DIGIT_RE           = re.compile(r"\d")
_SPECIAL_CHAR       = re.compile(r"[^a-zA-Z0-9\s]")


# ══════════════════════════════════════════════════════════════════════════════
#  EmailMetadataExtractor
# ══════════════════════════════════════════════════════════════════════════════
class EmailMetadataExtractor:
    """
    Extract a fixed-length feature vector from a raw email string.

    Usage
    -----
    >>> extractor = EmailMetadataExtractor()
    >>> features  = extractor.extract("URGENT! Click http://192.168.1.1/verify now!")
    >>> # returns dict with 12 named features
    """

    FEATURE_NAMES: List[str] = [
        "url_count",
        "has_ip_url",
        "exclamation_count",
        "uppercase_ratio",
        "urgent_word_count",
        "money_word_count",
        "avg_word_len",
        "char_count",
        "word_count",
        "digit_ratio",
        "special_char_count",
        "has_html_tags",
    ]

    def extract(self, text: str) -> Dict[str, float]:
        """Return a dict of 12 numeric features for one email string."""
        if not isinstance(text, str):
            text = str(text) if text else ""

        text_lower = text.lower()
        words      = text.split()
        n_words    = len(words)
        n_chars    = len(text)

        # 1. URL count
        urls = _URL_PATTERN.findall(text)
        url_count = len(urls)

        # 2. Has IP-based URL
        has_ip_url = float(bool(_IP_IN_URL.search(text)))

        # 3. Exclamation count
        exclamation_count = float(text.count("!"))

        # 4. Uppercase word ratio
        if n_words > 0:
            upper_words   = sum(1 for w in words if w.isupper() and len(w) > 1)
            uppercase_ratio = upper_words / n_words
        else:
            uppercase_ratio = 0.0

        # 5. Urgent word count
        words_lower = text_lower.split()
        urgent_word_count = float(
            sum(1 for w in words_lower if w.strip(".,!?;:\"'()") in URGENT_WORDS)
        )

        # 6. Money word count
        money_word_count = float(
            sum(1 for w in words_lower if w.strip(".,!?;:\"'()") in MONEY_WORDS)
        )

        # 7. Average word length
        if n_words > 0:
            avg_word_len = sum(len(w) for w in words) / n_words
        else:
            avg_word_len = 0.0

        # 8. Character count (log-scaled for large variance)
        char_count = float(math.log1p(n_chars))

        # 9. Word count (log-scaled)
        word_count = float(math.log1p(n_words))

        # 10. Digit ratio
        n_digits    = len(_DIGIT_RE.findall(text))
        digit_ratio = n_digits / max(n_chars, 1)

        # 11. Special character count
        special_char_count = float(len(_SPECIAL_CHAR.findall(text)))

        # 12. Has HTML tags
        has_html_tags = float(bool(_HTML_TAG.search(text)))

        return {
            "url_count"          : float(url_count),
            "has_ip_url"         : has_ip_url,
            "exclamation_count"  : exclamation_count,
            "uppercase_ratio"    : uppercase_ratio,
            "urgent_word_count"  : urgent_word_count,
            "money_word_count"   : money_word_count,
            "avg_word_len"       : avg_word_len,
            "char_count"         : char_count,
            "word_count"         : word_count,
            "digit_ratio"        : digit_ratio,
            "special_char_count" : special_char_count,
            "has_html_tags"      : has_html_tags,
        }

    def extract_vector(self, text: str) -> np.ndarray:
        """Return a 1D numpy array in the order of FEATURE_NAMES."""
        d = self.extract(text)
        return np.array([d[k] for k in self.FEATURE_NAMES], dtype=np.float64)


# ── Batch helper ─────────────────────────────────────────────────────────────
def extract_metadata_matrix(texts: List[str],
                             extractor: EmailMetadataExtractor = None
                            ) -> np.ndarray:
    """
    Process a list of email strings and return a 2D numpy matrix
    of shape (n_emails, n_features).

    Parameters
    ----------
    texts     : list of raw email strings
    extractor : EmailMetadataExtractor, optional (created if None)

    Returns
    -------
    np.ndarray  shape (n, 12)
    """
    if extractor is None:
        extractor = EmailMetadataExtractor()
    return np.vstack([extractor.extract_vector(t) for t in texts])


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "URGENT!!! Your account has been SUSPENDED. Click http://192.168.1.1/verify NOW!",
        "Hi John, please find the meeting notes attached. Best regards, Sarah.",
        "WIN a FREE iPhone! You are the lucky WINNER. Claim NOW at http://scam.xyz !!!",
        "Dear customer, your invoice for order #12345 is attached. No action required.",
    ]

    ext = EmailMetadataExtractor()
    print(f"{'Email (truncated)':<55}  {'URL':>4}  {'Excl':>5}  {'Urgnt':>6}  {'UpRt':>5}")
    print("-" * 80)
    for s in samples:
        feat = ext.extract(s)
        print(
            f"{s[:53]:<55}  "
            f"{feat['url_count']:>4.0f}  "
            f"{feat['exclamation_count']:>5.0f}  "
            f"{feat['urgent_word_count']:>6.0f}  "
            f"{feat['uppercase_ratio']:>5.3f}"
        )
