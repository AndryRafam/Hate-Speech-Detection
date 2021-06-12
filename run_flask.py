from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def index():
	return render_template('home.html')

@app.route("/Predict_Sentiment",methods=["POST","GET"])
def get_pred():
	max_words = 5000
	max_len = 200

	best_model3 = tf.keras.models.load_model("Model/best_model3.hdf5")
	sentiment3 = ['Hate Speech','Offensive','Neutral']

	with open('Main/tockenizer3.pickle','rb') as handle:
		tokenizer3 = pickle.load(handle)

	if request.method == "POST":
		text = request.form['sentiment']
		sequence = tokenizer3.texts_to_sequences([text])
		test = pad_sequences(sequence,maxlen=max_len)
		predictions_data3 = best_model3.predict(test)
		score_data3 = predictions_data3[0]
		res = sentiment3[np.around(best_model3(test),decimals=0).argmax(axis=1)[0]]
		acc = " ({:.2f} % of accuracy)".format(100*np.max(score_data3))
	#return redirect(url_for('home', result='baba'))
	return render_template('home.html',text=text,result=res+acc)

if __name__ == '__main__':
    app.run(debug= True)
