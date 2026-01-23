# helpfuncties.py
import pandas as pd

# functies importeren uit preprocessing.py module 
from preprocessing import(
    spacy_preprocessor,
    spacy_tokenizer,
)

def load_dataset(path="klachten.csv"):
    ''' Laad dataset in en verwijder dubbele klachten.'''
    df = pd.read_csv(path, index_col="ID") # ID als indexkolom
    df = df.drop_duplicates(subset="Omschrijving", keep="first") # Verwijder dubbele klachten op basis van Omschrijving kolom, behoud de eerste
    return df 

def get_all_tokens(df):
    ''' Haal alle tokens op uit de Omschrijving kolom.'''
    all_tokens = [] # lijst om alle tokens in op te slaan
    for text in df["Omschrijving"]: # loop over elke klachtomschrijving 
        all_tokens.extend(spacy_tokenizer(text)) # tokenize de tekst en voeg de tokens toe aan de lijst
    return all_tokens

def get_tokens_for_product(df, product):
    ''' Haal alle tokens op voor een specifiek productsector.'''
    subset = df[df["Product"] == product]["Omschrijving"] # filter de dataframe voor het specifieke product
    toks = [] # lijst om tokens in op te slaan  
    for text in subset: # loop over elke klachtomschrijving in de subset
        toks.extend(spacy_tokenizer(text)) # tokenize de tekst en voeg de tokens toe aan de lijst
    return toks
