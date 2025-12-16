import re
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

RE_WORD = re.compile(r"[a-zA-Z]{2,}")

def tokenize(text: str) -> list[str]:
    """
    Tokenize text into a list of normalized tokens:
    - lowercase
    - keep alphabetic tokens only (length >= 2)
    - remove English stopwords
    - NO stemming
    """
    if not text:
        return []

    text = text.lower()
    tokens = RE_WORD.findall(text)

    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens