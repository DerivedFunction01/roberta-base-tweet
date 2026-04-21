from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path


_QUOTE_EDGE_RE = re.compile(r'^[\s"\']+|[\s"\']+$')
_REPEATED_QUOTES_RE = re.compile(r'["\']{2,}')
_URL_RE = re.compile(
    r'(?:'
    r'https?://[^\s<>"\'()]+'
    r'|www\.[^\s<>"\'()]+'
    r'|[a-zA-Z0-9\-]+\.'
    r'(?:com|org|net|io|co|uk|edu|gov|me|dev|ai|app)'
    r'(?:/[^\s<>"\'()]*)?'
    r')',
    re.IGNORECASE,
)
_LITERAL_ESCAPE_RE = re.compile(r"\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|n|r|t)")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+")


def strip_quote_artifacts(text: str) -> str:
    """Remove obvious wrapper quotes and repeated quote artifacts."""
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = _REPEATED_QUOTES_RE.sub('"', text)
    text = _QUOTE_EDGE_RE.sub("", text)
    return text


def strip_links(text: str) -> str:
    """Remove obvious URL forms from tweet text."""
    return _URL_RE.sub(" ", text)


def normalize_literal_unicode(text: str) -> str:
    """Convert literal escaped unicode sequences into real text when present."""
    if not _LITERAL_ESCAPE_RE.search(text):
        return html.unescape(text)
    return html.unescape(text.encode("utf-8").decode("unicode_escape"))


@lru_cache(maxsize=1)
def load_dictionary_words() -> set[str]:
    """Load the NLTK dictionary word list used for optional uppercase normalization."""
    from nltk.corpus import words

    try:
        words.words()
    except LookupError:
        import nltk
        nltk.download('words')
    return {word.lower() for word in words.words() if word}


def lowercase_all_caps_dictionary_words(text: str, dictionary_words: set[str] | None = None) -> str:
    """Lowercase all-uppercase words when they are recognized dictionary words."""
    dictionary_words = dictionary_words if dictionary_words is not None else load_dictionary_words()
    pieces = []
    for token in _TOKEN_RE.findall(text):
        if token.isalpha() and token.isupper() and token.lower() in dictionary_words:
            pieces.append(token.lower())
        else:
            pieces.append(token)
    return "".join(pieces)


def clean_tweet_text(
    text: str,
    *,
    strip_quotes: bool = True,
    normalize_escapes: bool = True,
    lowercase_dictionary_caps: bool = False,
    dictionary_words: set[str] | None = None,
) -> str:
    """Apply the small tweet cleanup pipeline used for the first sentiment baseline."""
    cleaned = text or ""
    if normalize_escapes:
        cleaned = normalize_literal_unicode(cleaned)
    if strip_quotes:
        cleaned = strip_quote_artifacts(cleaned)
    cleaned = strip_links(cleaned)
    if lowercase_dictionary_caps:
        cleaned = lowercase_all_caps_dictionary_words(cleaned, dictionary_words=dictionary_words)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
