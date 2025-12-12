import spacy
import re
import pandas as pd

# Laad SpaCy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  

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
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if token.is_space:
            continue

        # verwijder xxx, XXXX etc.
        if re.fullmatch(r"x+", token.text, flags=re.IGNORECASE):
            continue

        lemma = token.lemma_.strip()
        if len(lemma) > 1:
            tokens.append(lemma)

    return tokens
