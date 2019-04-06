import pandas as pd
import os
import nltk,re
import string
import unicodedata
import numpy as np
import scipy as sp

from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer,SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word

pd.set_option('display.max_colwidth', 100)
# nltk.set_proxy('http://proxy.bloomberg.com:81')
pd.set_option('chained_assignment',None) 

# nltk.download('stopwords')

nb = MultinomialNB()

def get_all(folder):
    _all = os.listdir(folder)
    _all = [folder+'/' + i for i in _all]
    return _all

def extract_folder(folder):
	text_list = []
	for _file in get_all(folder):
		stri = open(_file, 'r',encoding="utf8").read()
		# stri = " ".join(stri.split()).translate(str.maketrans('', '', string.punctuation))
		stri = " ".join(stri.split())
		text_list.append(stri)
	return text_list

def reject_list():
	return set(extract_folder('output')) - set(extract_folder('pass'))


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        # print(word)
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def remove_numbers(words):
	no_numbers = []
	for word in words:
		no_number = ''.join([i for i in word if not i.isdigit()])
		no_numbers.append(no_number)
	return no_numbers

def remove_emails(paragraph):
	regex = r"\S*@\S*\s?"
	result = re.sub(regex, "", paragraph, 0)
	return result

def normalize(words):
    words = remove_non_ascii(words)
    # words = remove_emails(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    # words = replace_numbers(words)
    words = remove_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    return words

def tokenize_and_train(vect,X_train):
    print(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    print(('Features: ', X_train_dtm.shape[1]))
    return X_train_dtm

def transform(vect,X_test):
	X_test_dtm = vect.transform(X_test)
	return X_test_dtm

def train(X_train_dtm, y_train):
    nb.fit(X_train_dtm, y_train)
    return nb

def run_test(X_test_dtm):
    y_pred_class = nb.predict(X_test_dtm)
    # print(y_pred_class)
    return y_pred_class

def metric(y_test,X_test_dtm):
    from sklearn import metrics
    y_pred_class = run_test(X_test_dtm)
    print(('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class)))
    print(('Precision: ', metrics.precision_score(y_test, y_pred_class)))
    print(('Recall: ', metrics.recall_score(y_test, y_pred_class)))


def predict(matrix):
	result = nb.predict(matrix)
	return result


def norminalize_all(pass_df):
    for i in range(len(pass_df)):
	    a = remove_emails(pass_df["text"][i])
	    a = nltk.WhitespaceTokenizer().tokenize(a)
	    c = ' '.join(map(str, a))
	    c = nltk.tokenize.WordPunctTokenizer().tokenize(c)
	    c = normalize(c)
	    c = ' '.join(map(str, c))
	    e = nltk.WhitespaceTokenizer().tokenize(c)
	    e = ' '.join(map(str, e))
	    pass_df["text"][i] = e


def combine_pass_reject_df(pass_folder,reject_folder):
    pass_df = pd.DataFrame({"text":extract_folder(pass_folder),"passs":1})
    reject_df = pd.DataFrame({"text":extract_folder(reject_folder),"passs":0})
    pass_df = pass_df.append(reject_df, ignore_index=True)
    norminalize_all(pass_df)
    return pass_df

def new_resume_df(new_resume_folder):
    new_df = pd.DataFrame({"text":extract_folder(new_resume_folder)})
    norminalize_all(new_df)
    return new_df

def tf_idf_method(feature_set, target_set,random_state_value):
    X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, random_state=random_state_value)
    vect = TfidfVectorizer()
    X_train_dtm = tokenize_and_train(vect,X_train)
    nb = train(X_train_dtm, y_train)
    X_test_dtm = transform(vect,X_test)
    metric(y_test,X_test_dtm)
    X_new_dtm = transform(vect,X_new)
    print(predict(X_new_dtm))

def main():
    pass_df = combine_pass_reject_df('pass','reject')
    X_new = new_resume_df('new-resume')
    nb = tf_idf_method(pass_df['text'], pass_df['passs'], 20)   
    # X_test_dtm = transform(vect,X_test)
    # metric(y_test,X_test_dtm)
    # X_new_dtm = transform(vect,X_new)
    # print(predict(X_new_dtm))


if __name__ == "__main__":
    main()
