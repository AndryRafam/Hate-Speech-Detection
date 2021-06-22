import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

max_words = 5000
max_len = 200

best_model3 = tf.keras.models.load_model("../Model/best_model3.hdf5")
sentiment3 = ['Hate Speech','Offensive','Neither']

with open('tockenizer3.pickle','rb') as handle:
    tokenizer3 = pickle.load(handle)

print("\n")
string = input("=> ")
sequence = tokenizer3.texts_to_sequences([string])
test = pad_sequences(sequence,maxlen=max_len)
predictions_data3 = best_model3.predict(test)
score_data3 = predictions_data3[0]
print("\n")
print("{} ({:.2f} % of accuracy)".format(sentiment3[np.around(best_model3(test),decimals=0).argmax(axis=1)[0]],100*np.max(score_data3)))
print("\n")
