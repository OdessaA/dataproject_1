# helpfuncties.py
import pandas as pd

from preprocessing import(
    my_preprocessor,
    my_tokenizer,
)

def load_dataset(path="klachten.csv"):
    df = pd.read_csv(path).copy()
    df = df.drop(columns=["ID", "Datum_ontvangst", "Antwoord_bedrijf"])
    return df

def get_all_tokens(df):
    all_tokens = []
    for text in df["Omschrijving"]:
        all_tokens.extend(my_tokenizer(text))
    return all_tokens

def get_tokens_for_product(df, product):
    subset = df[df["Product"] == product]["Omschrijving"]
    toks = []
    for text in subset:
        toks.extend(my_tokenizer(text))
    return toks
