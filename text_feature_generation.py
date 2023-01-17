import pandas as pd
import numpy as np
import re
import nltk.corpus
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import cluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

def applyLemmatization(listOfTokens, stemmer):
    return [stemmer.lemmatize(token) for token in listOfTokens]

def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

def restrictLength(listOfTokens, lower = 2, upper = 21):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= lower or len(token) >= upper:
            twoLetterWord.append(token)
    return twoLetterWord

def processDataFrame(corpus):
    countries_list = [line.rstrip('\n').lower() for line in open('national_anthem_scrape/national_anthem_dataset/lists/countries.txt')]
    nationalities_list = [line.rstrip('\n') for line in open('national_anthem_scrape/national_anthem_dataset/lists/nationalities.txt')]
    other_words = [line.rstrip('\n') for line in open('national_anthem_scrape/national_anthem_dataset/lists/stopwords_scrapmaker.txt')]
    stopwords = nltk.corpus.stopwords.words('english')
    param_stemmer = SnowballStemmer('english')
    corpus = corpus.replace(",", "")
    corpus = corpus.rstrip('\n')
    corpus = corpus.lower()
    corpus = re.sub("\W_", " ", corpus)
    corpus = re.sub("\S*\d\S*", " ", corpus)
    corpus = re.sub("\S*@\S*\s?", " ", corpus)
    corpus = re.sub(r'http\S+', '', corpus)
    corpus = re.sub(r'www\S+', '', corpus)
    listOfTokens = word_tokenize(corpus)
    listOfTokens = removeWords(listOfTokens, stopwords)
    listOfTokens = removeWords(listOfTokens, countries_list)
    listOfTokens = removeWords(listOfTokens, nationalities_list)
    listOfTokens = removeWords(listOfTokens, other_words)
    listOfTokens = applyStemming(listOfTokens, param_stemmer)
    listOfTokens = removeWords(listOfTokens, other_words)
    
    corpus = " ".join(listOfTokens)
    corpus = unidecode(corpus)

    return corpus

data = pd.read_csv('national_anthem_scrape/national_anthem_dataset/anthems.csv')[['Country', 'Anthem']]
data['Country'] = data['Country'].str.capitalize()

corpus = data.apply(lambda row: processDataFrame(row['Anthem']), axis = 1).to_list()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
tf_idf = pd.DataFrame(data = X.toarray(), columns = vectorizer.get_feature_names_out())
final_df = tf_idf

print("{} rows".format(final_df.shape[0]))
print(final_df.T.nlargest(5,0))