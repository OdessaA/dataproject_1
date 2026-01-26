# Dataproject: Casus Klachteninstituut Financiële Dienstverlening

Dit project is uitgevoerd voor het Klachteninstituut Financiële Dienstverlening (KFID) in het kader van het vak Dataproject 1.  
Door aankomende Europese regelgeving verwacht KFID een sterke toename van het aantal klachten.

In dit project is onderzocht of klachtomschrijvingen automatisch kunnen worden geclassificeerd naar productsectoren, zodat KFID deze toename kan opvangen.

## Dataset
De dataset `klachten.csv`, die is ontvangen van de opdrachtgever, bevat 14.887 rijen en 5 kolommen en is volledig geanonimiseerd.  
Deze dataset wordt gebruikt voor de ontwikkeling van een automatisch tekstclassificatiemodel.

### Data Dictionary
![Data dictionary](afbeeldingen/DataDictionary.png)

## Bestandstructuur
In de map `afbeeldingen` staan de confusion matrices van alle getrainde modellen en de Data Dictionary. 
De map `data` bevat alle CSV-bestanden die binnen dit project zijn gebruikt of gegenereerd.

In `01_EDA.ipynb` wordt de Exploratory Data Analysis uitgevoerd.  
In `02_TextExploratie_Woordanalyse.ipynb` wordt een woordenanalyse uitgevoerd.  
In de notebooks `03` t/m `07` worden verschillende machine learning modellen getraind en geëvalueerd.

## Python-modules
Binnen het project zijn twee Python-modules aangemaakt:

- `helpfuncties.py`  
  Bevat functies voor het inladen van de dataset en het ophalen van tokens, zoals `load_dataset`, `get_all_tokens` en `get_tokens_for_product`.

- `preprocessing.py`  
  Bevat functies voor tekstpreprocessing met SpaCy, waaronder opschoning, tokenization en lemmatization.

## Environment
Voor dit project zijn twee afzonderlijke Python-omgevingen gebruikt.

### Klassieke modellen
Voor de klassieke machine-learning modellen (Logistic Regression, Naive Bayes, SGDClassifier en Random Forest) is gebruikgemaakt van:
- Python 3.13.9
- Besturingssysteem: Windows
- Requirements: requirements_klassiek.txt

### FinBERT 
Voor het FinBERT-model is een aparte omgeving opgezet vanwege de afhankelijkheid van PyTorch en Hugging Face:
- Python 3.12.10
- Besturingssysteem: Windows
- Requirements: requirements_finbert.txt

Deze omgeving bevat onder andere transformers, datasets, accelerate en PyTorch (met CUDA-ondersteuning) voor het trainen en evalueren van het FinBERT-model.