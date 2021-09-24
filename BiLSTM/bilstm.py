import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
import string
import random

from zipfile import ZipFile
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class Hate_Speech():
	
	def unzip(self,nm):
		with ZipFile(nm,"r") as zip:
			zip.extractall()

	def preprocess(self,y):
		self.x = y.lower()
		self.x = self.x.encode("ascii","ignore").decode()
		self.x = re.sub("https*\S+"," ",self.x)
		self.x = re.sub("@\S+"," ",self.x)
		self.x = re.sub("#\S+"," ",self.x)
		self.x = re.sub("\'\w+","",self.x)
		self.x = re.sub("[%s]" % re.escape(string.punctuation)," ", self.x)
		self.x = re.sub("\w*\d+\w*","",self.x)
		self.x = re.sub("\s{2,}"," ",self.x)
		return self.x

	def tokenize(self,y):
		for self.x in y:
			yield(word_tokenize(str(self.x)))

	def detokenize(self,txt):
		return TreebankWordDetokenizer().detokenize(txt)

	def model(self,inputs):
		self.x = Embedding(max_words,128)(inputs)
		self.x = Bidirectional(LSTM(64,return_sequences=True))(self.x)
		self.x = Bidirectional(LSTM(64))(self.x)
		self.outputs = Dense(3,activation="softmax")(self.x)
		self.model = Model(inputs,self.outputs)
		return self.model

ht = Hate_Speech()
	
ht.unzip("archive.zip")
df = pd.read_csv("hate_speech.csv")
temp = []
data_to_list = df["tweet"].values.tolist()
for i in range(len(data_to_list)):
	temp.append(ht.preprocess(data_to_list[i]))
			
data_words = list(ht.tokenize(temp))
	
final_data = []
for i in range(len(data_words)):
	final_data.append(ht.detokenize(data_words[i]))
print(final_data[:5])
final_data = np.array(final_data)

max_words = 20000
max_len = 200

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(final_data)
sequences = tokenizer.texts_to_sequences(final_data)
tweets = pad_sequences(sequences,maxlen=max_len)
with open("tokenizer.pickle","wb") as handle:
	pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
print(tweets)

labels = df["class"]

x_train,x_test,y_train,y_test = train_test_split(tweets,labels,random_state=42)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

model = ht.model(Input(shape=(None,),dtype="int32"))
model.summary()
model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])

checkpoint = ModelCheckpoint("hate_speech.h5",monitor="val_accuracy",save_best_only=True,save_weights_only=False)
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_val,y_val),callbacks=[checkpoint])
best = load_model("hate_speech.h5")
loss,acc = best.evaluate(x_test,y_test)
print("\nTest acc: {:.2f} %".format(100*acc))
print("Test loss: {:.2f} %".format(100*loss))