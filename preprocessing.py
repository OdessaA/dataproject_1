import pandas as pd 
import re 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from collections import Counter


stop_words = set(stopwords.words('english')) # initialiseren van stopwoorden met Engelse stopwoorden
lemmatizer = WordNetLemmatizer() # initialiseren van lemmatizer met WordNetLemmatizer

def preprocess_text(text):
    '''Preprocess tekst door te lowercasen en speciale tekens te verwijderen.'''
    if pd.isna(text): # als tekst ontbreekt, return lege string
        return"" 
    text = text.lower() # tekst lowercase maken
    text =re.sub(r'[^a-zA-Zà-ÿ\s]', ' ', text) # speciale tekens verwijderen 
    return text 

def my_tokenizer(text):
    '''Tokenize tekst, verwijder stopwoorden en lemmatize tokens.'''
    tokens = word_tokenize(text) # tokenizen
    tokens = [t for t in tokens if not re.fullmatch(r"x+", t.lower())] # verwijderen van tokens die alleen uit 'x' bestaan 
    tokens = [t for t in tokens if t not in stop_words] # verwijderen van stopwoorden
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # lemmatizen
    return tokens

def load_dataset(path='klachten.csv'):
    df = pd.read_csv(path).copy()
    df = df.drop(columns=['ID', 'Datum_ontvangst', 'Antwoord_bedrijf'])
    df = df.drop_duplicates(subset='Omschrijving', keep='first')
    return df 

def get_all_tokens(df):
    '''Functie die alle tokens teruggeeft'''
    all_tokens = [] # lijst om alle tokens op te slaan
    for text in df['Omschrijving']: 
        clean = preprocess_text(text) # tekst preprocessen
        tokens = my_tokenizer(clean) # tokenizen, stopwoorden verwijderen en lemmatizen
        all_tokens.extend(tokens) # tokens toevoegen aan lijst
    return all_tokens

def get_tokens_for_product(df, product):
    '''Functie die alle tokens voor een specifiek product teruggeeft'''
    subset = df[df['Product'] == product]['Omschrijving'] # subset voor specifiek product
    tokens = [] # lijst om tokens op te slaan
    for text in subset:
        clean = preprocess_text(text) # tekst preprocessen
        tokens.extend(my_tokenizer(clean)) # tokenizen, stopwoorden verwijderen en lemmatizen
    return tokens