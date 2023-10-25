from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import string
# pip install nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

def convert_to_lower(text):
    return text.lower()

 

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number




def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))



def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc


def prepare_data(path_to_data, encoding="latin-1"):
    data=pd.read_csv(path_to_data, encoding=encoding)

    data['label']=data['target'].map({'Not spam' : 0, 'spam': 1})
    data['messages'] = data['messages'].apply(lambda x: convert_to_lower(x))  
    data['messages'] = data['messages'].apply(lambda x: remove_numbers(x))
    data['messages'] = data['messages'].apply(lambda x: remove_punctuation(x))
    data['messages'] = data['messages'].apply(lambda x: remove_stopwords(x))

    data['messages'] = data['messages'].apply(lambda x: remove_extra_white_spaces(x))
    data['messages'] = data['messages'].apply(lambda x: lemmatizing(x))


    x=data['messages']
    y=data['label']

    return {'text':x,'label':y}



# Define the create_train_test_data function with TF-IDF and SMOTE
def create_train_test_data_with_smote(x, y, test_size, random_state):
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # Apply SMOTE to the training data
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    
    return {'x_train': x_train_resampled, 'x_test': x_test, 'y_train': y_train_resampled, 'y_test': y_test}, tfidf




