import pandas as pd
import numpy as np
import re
import nltk
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

def applyLemmatization(listOfTokens, stemmer):
    return [stemmer.lemmatize(token) for token in listOfTokens]

def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

def restrictLength(listOfTokens, lower = 3, upper = 21):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= lower or len(token) >= upper:
            twoLetterWord.append(token)
    return twoLetterWord

def processDataFrame(corpus, stemming = True):
    countries_list = [line.rstrip('\n').lower() for line in open('national_anthem_scrape/national_anthem_dataset/lists/countries.txt')]
    nationalities_list = [line.rstrip('\n') for line in open('national_anthem_scrape/national_anthem_dataset/lists/nationalities.txt')]
    other_words = [line.rstrip('\n') for line in open('national_anthem_scrape/national_anthem_dataset/lists/stopwords_scrapmaker.txt')]
    stopwords = nltk.corpus.stopwords.words('english')
    if stemming == True:
        param_stemmer = SnowballStemmer('english')
    else:
        param_stemmer = WordNetLemmatizer()
    corpus = corpus.replace(",", "")
    corpus = corpus.rstrip('\n')
    corpus = corpus.lower()
    
    # REGEX SUBSTITUTION
    corpus = re.sub("\W_", " ", corpus)
    corpus = re.sub("\S*\d\S*", " ", corpus)
    corpus = re.sub("\S*@\S*\s?", " ", corpus)
    corpus = re.sub(r'http\S+', '', corpus)
    corpus = re.sub(r'www\S+', '', corpus)
    
    # TOKENIZE TEXT DATA
    listOfTokens = word_tokenize(corpus)
    
    # REMOVE STOPWORDS AND COUNTRY SPECIFIC REFERENCES
    listOfTokens = removeWords(listOfTokens, stopwords)
    listOfTokens = removeWords(listOfTokens, countries_list)
    listOfTokens = removeWords(listOfTokens, nationalities_list)
    listOfTokens = removeWords(listOfTokens, other_words)

    if stemming:
        listOfTokens = applyStemming(listOfTokens, param_stemmer)
    else:
        listOfTokens = applyLemmatization(listOfTokens, param_stemmer)

    twoLetterWord = restrictLength(listOfTokens)
    listOfTokens = removeWords(listOfTokens, twoLetterWord)

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
n_largest = 30
cols_of_interest = final_df.T.nlargest(n_largest, 0).index
cols_of_interest = [y for x in [["Country"], cols_of_interest] for y in x]
final_df['Country'] = data['Country']
final_df = final_df[cols_of_interest]
print(cols_of_interest)