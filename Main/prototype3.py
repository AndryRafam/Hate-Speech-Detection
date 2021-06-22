import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
print("Done")

train = pd.read_csv("../Data/data.csv")
print(train.head(15))
print(len(train))
print(train.groupby('class').nunique())
train = train[['tweet','class']]
print(train.head())
print(train['tweet'].isnull().sum())
stop_words = stopwords.words("english")
wordnet = WordNetLemmatizer()

# Data processing
def text_preproc(x):
	x = x.lower()
	x = " ".join([word for word in x.split(" ") if word not in stop_words])
	x = x.encode("ascii", "ignore").decode()
	x = re.sub(r"https*\S+", " ", x)
	x = re.sub(r"@\S+", " ", x)
	x = re.sub(r"#\S+", " ", x)
	x = re.sub(r"\",\w+", "", x)
	x = re.sub("[%s]" % re.escape(string.punctuation), " ", x)
	x = re.sub(r"\w*\d+\w*", "", x)
	x = re.sub(r"\s{2,}", " ", x)
	return x
	
final_data = []
data_to_list = train["tweet"].values.tolist()
for i in range(len(data_to_list)):
	final_data.append(text_preproc(data_to_list[i]))
print(list(final_data[:5]))

final_data = np.array(final_data)

labels = np.array(train["class"])
labels = tf.keras.utils.to_categorical(labels,3,dtype="int32")
print(len(labels))

# Data Sequencing and Splitting
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

max_words = 5000 # Consider only the top 5k words
max_len = 200 # Consider only the 200 first word of each tweet

tokenizer3 = Tokenizer(num_words=max_words)
tokenizer3.fit_on_texts(final_data)
sequences = tokenizer3.texts_to_sequences(final_data)
tweets = pad_sequences(sequences, maxlen=max_len)
with open('tockenizer3.pickle','wb') as handle:
    pickle.dump(tokenizer3,handle,protocol=pickle.HIGHEST_PROTOCOL)

print(tweets)
print(labels)

x_train, x_test, y_train, y_test = train_test_split(tweets,labels,random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
print(len(x_train),len(x_val),len(x_test),len(y_train),len(y_val),len(y_test))

# Building and train NN
## BiDR LST LAYER MODEL
model = Sequential([
    layers.Embedding(max_words,128,input_length=max_len),
    layers.Bidirectional(layers.LSTM(64,return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(3,activation="softmax"),
])
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
checkpoint = ModelCheckpoint("../Model/best_model3.hdf5", save_best_only=True, save_weights_only=False)
history = model.fit(x_train, y_train, epochs=4, validation_data=(x_val,y_val), callbacks=[checkpoint])

# Test the model on X_test
best_model = tf.keras.models.load_model("../Model/best_model3.hdf5")
test_loss, test_acc = best_model.evaluate(x_test,y_test,verbose=2)
print("Test accuracy: {:.2f}%".format(100*test_acc))
