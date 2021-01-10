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

df = pd.read_csv('https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv')
# df.head()
# Shape of the dataframe
# df.shape

needed_columns = ['user_name','date','text']
df = df[needed_columns]
# df.shape()

# Changing category of the column
df.user_name = df.user_name.astype('category')
df.user_name = df.user_name.cat.codes # Assigns unique numerrical value for each unique user name

df.date = pd.to_datetime(df.date).dt.date
# df.head()

# Tweet's texts
texts = df['text']

# Removing URLs from text
remove_url = lambda x: re.sub(r'https\S+', '', str(x))
texts_lr = texts.apply(remove_url)

# Converting tweets to lowercase
to_lower = lambda x: x.lower()
texts_lr_lc = texts_lr.apply(to_lower)

# Removing punctuations
remove_puncs = lambda x: x.translate(str.maketrans('', '', string.punctuation))
texts_lr_lc_np = texts_lr_lc.apply(remove_puncs)