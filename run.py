import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import re
import string

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PosterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('vader_lexicon')

from collections import Counter

import seaborn as sns
import plotly.express as px

sns.set(style='darkgrid')

df = pd.read_csv('https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv') # df - data frame
# df.head()
# Shape of the dataframe
# df.shape

needed_columns = ['user_name','date','text']
df = df[needed_columns]
# df.head()

# Changing category of the column
df.user_name = df.user_name.astype('category')
df.user_name = df.user_name.cat.codes # Assigns unique numerrical value for each unique user name

df.date = pd.to_datetime(df.date).dt.date
# df.head()

# Tweet's texts
texts = df['text']

# Removing URLs from text
remove_url = lambda x: re.sub(r'https\S+', '', str(x))
texts_lr = texts.apply(remove_url) # lr - link removed

# Converting tweets to lowercase
to_lower = lambda x: x.lower()
texts_lr_lc = texts_lr.apply(to_lower) # lc - lowercase

# Removing punctuations
remove_puncs = lambda x: x.translate(str.maketrans('', '', string.punctuation))
texts_lr_lc_np = texts_lr_lc.apply(remove_puncs) # np - no punctuation

# Removing stopwords
covid_words = ['covid', '#coronavirus', '#coronavirusoutbrake', '#coronavirusPandemic',
               '#covid19', '#covid_19', '#epitwitter', '#ihavecorona',
               'amp', 'coronavirus', 'covid19']
stop_words = set(stopwords.words('English'))
stop_words.update(covid_words)
remove_words = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
texts_lr_lc_np_ns = texts_lr_lc_np.apply(remove_words) # ns - no stopwords

# Creating list of words out of all the tweets
words_list = [word for line in texts_lr_lc_np_ns for word in line.split()]
word_counts = Counter(words_list).most_common(50)
words_df = pd.DataFrame(word_counts)
words_df.columns = ['Word', 'Frequency']
# px.bar(words_df, x='Word', y='Frequency', title='Most common word') # Bar chart with 50 most commonly used words

# Assigning preProcessed text to the main data frame
df.text = texts_lr_lc_np_ns