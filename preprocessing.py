#preprocessing.py
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Stopwoorden + lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def my_preprocessor(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def my_tokenizer(text):
    tokens = word_tokenize(my_preprocessor(text))
    tokens = [t for t in tokens if t not in stop_words]
    
    # verwijder tokens zoals "x", "xx", "xxx", "xxxx", in lowercase of uppercase
    tokens = [t for t in tokens if not re.fullmatch(r"x+", t, flags=re.IGNORECASE)]
    
    return [lemmatizer.lemmatize(t, pos="v") for t in tokens]
