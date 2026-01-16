# preprocessing.py
import re
import pandas as pd

# Laad globale SpaCy NLP model variabele
_nlp = None

def get_nlp():
    ''' Laad en retourneer het SpaCy NLP model.'''
    global _nlp # gebruik de globale variabele _nlp
    if _nlp is None: # als het model nog niet is geladen
        import spacy # importeer spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"]) # laad het Engelse model zonder NER en parser voor snellere verwerking
    return _nlp


def spacy_preprocessor(text):
    ''' Basis tekstvoorverwerking met SpaCy.'''
    if pd.isna(text): # Als de tekst leeg (NaN) is, geef een lege string terug
        return ""
    
    # lowercase + basis schoonmaak
    text = text.lower() 
    text = re.sub(r"[^\w\s]+", " ", text) # Alles behalve letters, cijfers en spaties wordt vervangen door een spatie
    text = re.sub(r"\d+", " ", text) # Verwijder cijfers
    text = re.sub(r"\s+", " ", text).strip() # Vervang meerdere spaties door één spatie en strip begin/einde
    return text

def spacy_tokenizer(text):
    ''' Tokenizer met SpaCy inclusief lemmatization en filtering.'''
    text = spacy_preprocessor(text) # Basis preprocessing toepassen
    doc = get_nlp()(text) # verwerk de tekst met SpaCy tot een Doc object

    tokens = [] # lijst om de tokens in op te slaan

    # Loop over alle tokens die SpaCy herkent 
    for token in doc:
        if token.is_stop: # stopwoorden overslaan
            continue
        if token.is_punct: # leestekens overslaan
            continue
        if token.is_space: # spaties overslaan
            continue

        if re.fullmatch(r"x+", token.text, flags=re.IGNORECASE): # Verwijder tokens die alleen uit x'en bestaan (bijv. "xx", "xxx")
            continue

        lemma = token.lemma_.strip() # gebruik de lemma-vorm van het token (basisvorm)
        if len(lemma) > 1: # alleen lemmas met meer dan 1 karakter behouden om ruis te verminderen
            tokens.append(lemma)

    return tokens

def minimal_clean(text):
    ''' Minimale schoonmaak van tekst.'''
    if pd.isna(text):
        return ""
    
    text = re.sub(r"\b[xX]{2,}\b", " ", text) # verwijder tokens die alleen uit x bestaan
    text = re.sub(r"\s+", " ", text).strip() # Vervang meerdere spaties door één spatie en strip begin/einde
    return text
