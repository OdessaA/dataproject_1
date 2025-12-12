# helpfuncties.py
import pandas as pd

from preprocessing import(
    spacy_preprocessor,
    spacy_tokenizer,
)

def load_dataset(path="klachten.csv"):
    df = pd.read_csv(path).copy()
    df = df.drop(columns=["ID", "Datum_ontvangst", "Antwoord_bedrijf"])
    df = df.drop_duplicates(subset='Omschrijving', keep='first')
    return df

def get_all_tokens(df):
    all_tokens = []
    for text in df["Omschrijving"]:
        all_tokens.extend(spacy_tokenizer(text))
    return all_tokens

def get_tokens_for_product(df, product):
    subset = df[df["Product"] == product]["Omschrijving"]
    toks = []
    for text in subset:
        toks.extend(spacy_tokenizer(text))
    return toks
