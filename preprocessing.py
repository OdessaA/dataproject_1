import re
import pandas as pd

# Laad SpaCy
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _nlp


def spacy_preprocessor(text):
    if pd.isna(text):
        return ""
    
    # lowercase + basis schoonmaak
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def spacy_tokenizer(text):
    text = spacy_preprocessor(text)
    doc = get_nlp()(text)

    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if token.is_space:
            continue

        if re.fullmatch(r"x+", token.text, flags=re.IGNORECASE):
            continue

        lemma = token.lemma_.strip()
        if len(lemma) > 1:
            tokens.append(lemma)

    return tokens

def minimal_clean(text):
    if pd.isna(text):
        return ""
    # verwijder tokens die alleen uit x bestaan
    text = re.sub(r"\b[xX]{2,}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
