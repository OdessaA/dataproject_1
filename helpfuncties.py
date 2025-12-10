# helpfuncties.py
import pandas as pd

from preprocessing import(
    load_dataset,
    my_preprocessor,
    my_tokenizer,
)

def load_dataset(path="klachten.csv"):
    """Dataset laden + onnodige kolommen verwijderen + duplicates."""
    df = pd.read_csv(path).copy()
    df = df.drop(columns=["ID", "Datum_ontvangst", "Antwoord_bedrijf"])
    return df


def get_all_tokens(df):
    """Alle tokens uit hele dataset verzamelen."""
    all_tokens = []
    for text in df["Omschrijving"]:
        all_tokens.extend(my_tokenizer(text))
    return all_tokens


def get_tokens_for_product(df, product):
    """Tokens voor één specifieke productklasse."""
    subset = df[df["Product"] == product]["Omschrijving"]
    toks = []
    for text in subset:
        toks.extend(my_tokenizer(text))
    return toks