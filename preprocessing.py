import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from scipy import sparse
from sklearn.preprocessing import FunctionTransformer


def _ensure_nltk_resource(resource: str, path: str) -> None:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)


_ensure_nltk_resource('punkt', 'tokenizers/punkt')
_ensure_nltk_resource('stopwords', 'corpora/stopwords')
_ensure_nltk_resource('wordnet', 'corpora/wordnet')

BASIS_STOPWOORDEN = set(stopwords.words('english'))
AANGEPASTE_STOPWOORDEN = {
    'complaint', 'complaints', 'customer', 'company', 'department',
    'dear', 'sir', 'madam', 'hello', 'regarding', 'subject',
    'case', 'issue', 'thanks', 'thank', 'please', 'sincerely',
    'accountnumber', 'team'
}
STOP_WOORDEN = BASIS_STOPWOORDEN.union(AANGEPASTE_STOPWOORDEN)
LEMMATISATOR = WordNetLemmatizer()
NORMALISATIES = {
    r'\bcc\b': 'creditcard',
    r'\bcc\.': 'creditcard',
    r'\bacct\b': 'account',
    r'\bacct\.': 'account',
    r'\bach\b': 'automatedclearinghouse',
    r'\btxn\b': 'transaction',
    r'\bchk\b': 'checking',
}
SLEUTELWOORDEN = {
    'kredietregistratie': [
        'equifax',
        'experian',
        'transunion',
        'bureau',
        'tradeline',
        'inquiry',
        'public record',
        'credit score',
        'credit file',
        'credit reporting',
        'reinvestigation',
        'furnisher',
    ],
    'incasso': [
        'collection',
        'collection agency',
        'debt collector',
        'collection account',
        'garnishment',
        'harassment',
        'settlement',
        'validation',
        'threatening',
        'third party',
    ],
    'hypotheek': [
        'mortgage',
        'foreclosure',
        'escrow',
        'escrow account',
        'property',
        'short sale',
        'loan modification',
        'home loan',
        'appraisal',
        'closing',
        'refinance',
        'reverse mortgage',
    ],
    'creditcard': [
        'credit card',
        'cardmember',
        'charge',
        'transaction',
        'purchase',
        'merchant',
        'chargeback',
        'swipe',
        'chip',
        'card limit',
        'annual fee',
        'cashback',
    ],
    'consumentenkrediet': [
        
        'vehicle',
        'auto',
        'car',
        'dealership',
        'auto finance',
        'deficiency balance',
        'repossession',
        'trade-in',
        'odometer',
        'balloon payment',
        'extended warranty',
    ],
    'bankrekening': [
        'checking',
        'savings',
        'deposit',
        'overdraft',
        'direct deposit',
        'atm',
        'withdrawal',
        'wire',
        'transfer',
        'branch',
        'teller',
        'debit card',
    ],
}



def preprocess_text(tekst: str) -> str:
    if pd.isna(tekst):
        return ''
    if not isinstance(tekst, str):
        tekst = str(tekst)
    tekst = tekst.lower()
    for patroon, vervanging in NORMALISATIES.items():
        tekst = re.sub(patroon, vervanging, tekst)
    tekst = re.sub(r'[^a-z\s]', ' ', tekst)
    tekst = re.sub(r'\s+', ' ', tekst).strip()
    return tekst


def my_tokenizer(tekst: str):
    tokens = word_tokenize(tekst)
    tokens = [tok for tok in tokens if tok not in STOP_WOORDEN]
    tokens = [LEMMATISATOR.lemmatize(tok) for tok in tokens]
    return tokens


def load_dataset(path='klachten.csv', include_answer=False):
    df = pd.read_csv(path).copy()
    kolommen = ['ID', 'Datum_ontvangst']
    if 'Antwoord_bedrijf' in df.columns and not include_answer:
        kolommen.append('Antwoord_bedrijf')
    
    df = df.drop(columns = [kol for kol in kolommen if kol in df.columns])
    df = df.drop_duplicates(subset='Omschrijving', keep='first')
    return df


def get_all_tokens(df):
    all_tokens = []
    for text in df['Omschrijving']:
        clean = preprocess_text(text)
        tokens = my_tokenizer(clean)
        all_tokens.extend(tokens)
    return all_tokens


def get_tokens_for_product(df, product):
    subset = df[df['Product'] == product]['Omschrijving']
    tokens = []
    for text in subset:
        clean = preprocess_text(text)
        tokens.extend(my_tokenizer(clean))
    return tokens


def keyword_indicator(teksten):
    serie = pd.Series(teksten).fillna('').str.lower()
    kolommen = []
    for woorden in SLEUTELWOORDEN.values():
        patroon = r'(?:' + '|'.join(re.escape(w) for w in woorden) + r')'
        kolommen.append(serie.str.count(patroon).to_numpy(dtype=float))
    if not kolommen:
        return sparse.csr_matrix((len(serie), 0))
    matrix = np.column_stack(kolommen)
    return sparse.csr_matrix(matrix)


def extra_kenmerken(teksten):
    serie = pd.Series(teksten).fillna('')
    lengtes = serie.str.len().to_numpy(dtype=float).reshape(-1, 1)
    hoofdletters = serie.str.count(r'[A-Z]').to_numpy(dtype=float).reshape(-1, 1)
    cijfers = serie.str.count(r'\d').to_numpy(dtype=float).reshape(-1, 1)
    uitroeptekens = serie.str.count(r'!').to_numpy(dtype=float).reshape(-1, 1)
    ratios = np.divide(
        hoofdletters,
        lengtes + 1e-6,
        out=np.zeros_like(hoofdletters),
        where=lengtes != 0
    )
    kenmerken = np.hstack([lengtes, hoofdletters, cijfers, uitroeptekens, ratios])
    return sparse.csr_matrix(kenmerken)


keyword_transformer = FunctionTransformer(keyword_indicator, validate=False)
extra_transformer = FunctionTransformer(extra_kenmerken, validate=False)
