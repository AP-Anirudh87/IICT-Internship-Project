"""
========================================================
  Fake News Detection – Text Preprocessor
  IICT Summer Internship Project 1
========================================================
Manual implementation of text cleaning and tokenization –
no NLTK or spaCy required.  Includes:
  • HTML tag / punctuation removal
  • Lowercasing
  • Numeric token removal
  • Stopword filtering (hardcoded standard English list)
  • White-space tokenization
"""

import re
from typing import List


# ── English Stopwords (hardcoded, no external dependency) ────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "because", "as", "until", "while", "although", "though",
    "even", "s", "t", "re", "ve", "ll", "d", "m", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "about", "between", "same", "own", "against",
    "only", "also", "among", "said",
}


# ── Core Cleaner ─────────────────────────────────────────────────────────────
class TextPreprocessor:
    """
    Stateless text cleaning and tokenization pipeline.

    Usage
    -----
    >>> tp = TextPreprocessor()
    >>> tokens = tp.tokenize("Breaking NEWS! Visit http://example.com/story")
    >>> tokens
    ['breaking', 'news', 'visit']
    """

    # Compiled patterns (compiled once at class creation for speed)
    _HTML_TAG    = re.compile(r"<[^>]+>")
    _URL         = re.compile(r"https?://\S+|www\.\S+")
    _EMAIL       = re.compile(r"\S+@\S+")
    _PUNCT       = re.compile(r"[^a-zA-Z\s]")
    _WHITESPACE  = re.compile(r"\s+")

    def __init__(self,
                 remove_stopwords: bool = True,
                 min_token_len:    int  = 2) -> None:
        self.remove_stopwords = remove_stopwords
        self.min_token_len    = min_token_len

    # ── Private helpers ──────────────────────────────────────────────────────
    def _strip_html(self, text: str) -> str:
        return self._HTML_TAG.sub(" ", text)

    def _strip_urls(self, text: str) -> str:
        return self._URL.sub(" ", text)

    def _strip_emails(self, text: str) -> str:
        return self._EMAIL.sub(" ", text)

    def _strip_punctuation(self, text: str) -> str:
        return self._PUNCT.sub(" ", text)

    def _normalize_whitespace(self, text: str) -> str:
        return self._WHITESPACE.sub(" ", text).strip()

    # ── Public API ───────────────────────────────────────────────────────────
    def clean(self, text: str) -> str:
        """Return a cleaned, lowercase plain-text string."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = self._strip_html(text)
        text = self._strip_urls(text)
        text = self._strip_emails(text)
        text = self._strip_punctuation(text)
        text = text.lower()
        text = self._normalize_whitespace(text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Clean and tokenize text into a list of word tokens."""
        cleaned = self.clean(text)
        tokens  = cleaned.split()
        tokens  = [t for t in tokens if len(t) >= self.min_token_len]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def process(self, text: str) -> str:
        """Return cleaned, tokenized text joined as a single string.
        Suitable as input to sklearn's TfidfVectorizer.
        """
        return " ".join(self.tokenize(text))


# ── Batch helper ─────────────────────────────────────────────────────────────
def preprocess_series(texts, preprocessor: TextPreprocessor = None) -> List[str]:
    """
    Process an iterable (e.g. a pandas Series) of texts.

    Parameters
    ----------
    texts : iterable of str
    preprocessor : TextPreprocessor, optional
        If None, a default TextPreprocessor is created.

    Returns
    -------
    List[str]  – cleaned/tokenized text strings, one per input.
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    return [preprocessor.process(t) for t in texts]


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Breaking NEWS! The president <b>said</b> visit https://example.com/foo",
        "URGENT: You are a WINNER! Click here now!!!  123",
        "   Multiple    spaces  and   HTML <p>tags</p> removed.   ",
        None,
        "",
    ]
    tp = TextPreprocessor()
    print(f"{'Original':<60}  →  Cleaned tokens")
    print("-" * 100)
    for s in samples:
        tokens = tp.tokenize(str(s))
        print(f"{str(s)[:58]:<60}  →  {tokens}")
