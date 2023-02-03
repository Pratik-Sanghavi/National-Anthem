import re
import pandas as pd
import nltk
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class TextFeatureGeneration():
    def __init__(self,
                 data_path = 'national_anthem_scrape/national_anthem_dataset/anthems.csv',
                 country_column = 'Country',
                 anthem_column = 'Anthem',
                 countries_list_path = 'national_anthem_scrape/national_anthem_dataset/lists/countries.txt',
                 nationalities_list_path = 'national_anthem_scrape/national_anthem_dataset/lists/nationalities.txt',
                 other_words_list_path = 'national_anthem_scrape/national_anthem_dataset/lists/stopwords_scrapmaker.txt'
                 ):

        self.data_path = data_path
        self.country_col = country_column
        self.anthem_col = anthem_column
        self.countries_list_path = countries_list_path
        self.nationalities_list_path = nationalities_list_path
        self.other_words_list_path = other_words_list_path
        if countries_list_path:
            self.countries_list = [line.rstrip('\n').lower() for line in open(self.countries_list_path)]
        if nationalities_list_path:
            self.nationalities_list = [line.rstrip('\n') for line in open(self.nationalities_list_path)]
        if other_words_list_path:
            self.other_words = [line.rstrip('\n') for line in open(self.other_words_list_path)]
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.feature_df = None

    def removeWords(self, listOfTokens, listOfWords):
        return [token for token in listOfTokens if token not in listOfWords]
    
    def applyLemmatization(self, listOfTokens, stemmer):
        return [stemmer.lemmatize(token) for token in listOfTokens]

    def applyStemming(self, listOfTokens, stemmer):
        return [stemmer.stem(token) for token in listOfTokens]

    def restrictLength(self, listOfTokens, lower = 3, upper = 21):
        twoLetterWord = []
        for token in listOfTokens:
            if len(token) <= lower or len(token) >= upper:
                twoLetterWord.append(token)
        return twoLetterWord

    def processDataFrame(self, corpus, stemming = True):
        if stemming == True:
            param_stemmer = SnowballStemmer('english')
        else:
            param_stemmer = WordNetLemmatizer()
        
        # BASIC OPERATIONS
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
        listOfTokens = self.removeWords(listOfTokens, self.stopwords)
        listOfTokens = self.removeWords(listOfTokens, self.countries_list)
        listOfTokens = self.removeWords(listOfTokens, self.nationalities_list)
        listOfTokens = self.removeWords(listOfTokens, self.other_words)

        if stemming:
            listOfTokens = self.applyStemming(listOfTokens, param_stemmer)
        else:
            listOfTokens = self.applyLemmatization(listOfTokens, param_stemmer)

        twoLetterWord = self.restrictLength(listOfTokens)
        listOfTokens = self.removeWords(listOfTokens, twoLetterWord)

        corpus = " ".join(listOfTokens)
        corpus = unidecode(corpus)

        return corpus

    def textFeaturization(self, frac = 0.1, max_features = 100):
        data = pd.read_csv(self.data_path)[[self.country_col, self.anthem_col]]
        data[self.country_col] = data[self.country_col].str.capitalize()
        data[self.anthem_col] = data[self.anthem_col].astype(str)
        
        corpus = data.apply(lambda row: self.processDataFrame(row[self.anthem_col], False), axis = 1).to_list()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        final_df = pd.DataFrame(data = X.toarray(), columns = vectorizer.get_feature_names_out())
        n_largest = min(int(frac*len(vectorizer.get_feature_names_out())), max_features)
        cols_of_interest = final_df.sum(axis = 0).sort_values(ascending = False).index.tolist()[:n_largest]
        cols_of_interest = [y for x in [["Country_Name"], cols_of_interest] for y in x]

        final_df['Country_Name'] = data[self.country_col]
        final_df = final_df[cols_of_interest]
        self.feature_df = final_df

    def writeFile(self, out_file_path):
        if self.feature_df is None:
            print('Run the textFeaturization method first!')
        self.feature_df.to_csv(out_file_path, index = False)