import tensorflow as tf
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import string
import nltk
import gensim
import spacy
import pickle
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
print('Done')

train = pd.read_csv("Data/data.csv")
print(train.head(15))
print(len(train))
print(train.groupby('class').nunique())
train = train[['tweet','class']]
print(train.head())
print(train['tweet'].isnull().sum())

# Data processing
def depure_data(data):
    #Removing URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'',data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    return data

temp = []
# Splitting pd.Series to data to list
data_to_list = train["tweet"].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
print(list(temp[:5]))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(sent_to_words(temp))
print(data_words[:10])
print(len(data_words))

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
print(data[:5])

data = np.array(data)

# Labels
labels = np.array(train["class"])
labels = tf.keras.utils.to_categorical(labels,3,dtype="int32")
print(len(labels))

# Data Sequencing and Splitting
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
print('Done')

max_words = 5000 # Consider only the top 5k words
max_len = 200 # Consider only the 200 first word of each tweet

tokenizer3 = Tokenizer(num_words=max_words)
tokenizer3.fit_on_texts(data)
sequences = tokenizer3.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
with open('tockenizer3.pickle','wb') as handle:
    pickle.dump(tokenizer3,handle,protocol=pickle.HIGHEST_PROTOCOL)

print(tweets)
print(labels)

x_train, x_test, y_train, y_test = train_test_split(tweets,labels,random_state=0)
print(len(x_train),len(x_test),len(y_train),len(y_test))

# Building and train NN
## BiDR LST LAYER MODEL
model = Sequential([
    layers.Embedding(max_words,128,input_length=max_len),
    layers.Bidirectional(layers.LSTM(64,return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(3,activation="softmax"),
])
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
checkpoint = ModelCheckpoint("best_model3.hdf5",monitor="val_accuracy",verbose=1,save_best_only=True,mode='auto',period=1,save_weights_only=False)
history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test,y_test), verbose=2, callbacks=[checkpoint])

# Test the model on dataset X_test
best_model = tf.keras.models.load_model("best_model3.hdf5")
test_loss, test_acc = best_model.evaluate(x_test,y_test,verbose=2)
print("Test accuracy: {:.2f}%".format(100*test_acc))
