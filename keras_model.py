# These are the premade imports
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# There are our imports
from web_scraper import create_data
import pickle
from utils import *
"""
So first, we need two inputs, an image with size 60*60 and a vector of 10 random 
numbers (the gaussian distribution) for additional variety. After a certain 
number of layers, the hidden layers of both become the same length. Then we 
concatenate them and send them into an LSTM which produces an output.
"""

#hyperparameters
add_noise = False
learning_rate = 0.01
optimizer = Adam(learning_rate)
meme_phrase_per_meme = 15

input_image_shape = (60,60,1)
input_noise_size = 100
input_words_image = 50 # the max was 24 words in the test data
output_vector_size = 823
embedding_dim = 200
from web_scraper import *

'''
Creating the data generator (since we are using hella memory, we need 
a generator)
'''
def data_creator(phrases, memes, word2idx, num_photos_per_batch):
	# initializes the lists which we will later yield
	X_image, X_noise, X_captions, y_values = [], [], [], []
	# creates the counter to get the the num_photos_per_batch before resetting
	n = 0
	while True:
		# for each meme (there are 15 meme phrases per meme)
		for index in range(0, len(memes), meme_phrase_per_meme):
			# the phrases for this meme = meme_phrases
			meme_phrases = phrases[index//meme_phrase_per_meme]
			# for each phrase in the meme_phrases
			for meme_phrase in meme_phrases:

				# you are making a list of numbers relating to the words
				seq = [word2idx[word] for word in meme_phrase.split(' ') if word in word2idx]
				sample_z_value = sample_z(1, input_noise_size)[0]

				# for each of the possible sub_sequences in the sequence of numbers
				for i in range(len(seq)):
					# the input sequence is the current words for the LSTM
					# the output word is going to be the y output for the model
					input_sequences, output_word = seq[:i], seq[i]
					# This adds extra zeroes so the LSTM is always the same length
					input_sequences = pad_sequences(input_sequences, input_words_image)
					# This takes in the number value of the output and makes a list of 0s and one 1
					# that could be used as the label in the network
					output_word = categorize_variable(output_word, output_vector_size)
					# this adds the Xs and ys to the list
					X_image.append(memes[index])
					X_noise.append(sample_z_value)
					X_captions.append(input_sequences)
					y_values.append(output_word)
					# increment n for every test example
					n+=1 # <-- you can probably sub this in for len(X_image) when need be
				# if the number of test examples in the current is equal to the limit
				if n >= num_photos_per_batch:
					# this yields the batch of Xs and ys
					if add_noise:
						yield ([np.array(X_image), np.array(X_noise), np.array(X_captions)], np.array(y_values))
					else:
						yield ([np.array(X_image), np.array(X_captions)], np.array(y_values))
					# this resets the values
					n = 0
					X_image, X_noise, X_captions, y_values = [], [], [], []

'''
Creating the model. It takes in three inputs, one image of size (60,60,1), one
noise vector of the length described in the hyperparamters (input_noise_size), and a visible_input of
size described in the hyperparamters (input_words_image)
'''

def model_creator():
	visible_image = Input(input_image_shape)
	flattened = Flatten()(visible_image)
	first_image_hidden = Dense(256, activation = "relu")(flattened)
	first_image_hidden = Dropout(0.5)(first_image_hidden)
	second_image_hidden = Dense(128, activation = "relu")(first_image_hidden)

	if add_noise:
		visible_noise = Input((input_noise_size,))
		first_noise_hidden = Dense(128, activation = "relu")(visible_noise)

	visible_input = Input((input_words_image,))
	embeddings = Embedding(output_vector_size, embedding_dim, mask_zero=True)(visible_input)
	embeddings = Dropout(0.5)(embeddings)
	LSTM_hidden_layer = LSTM(128)(embeddings)

	if not add_noise:
		concatenated = Concatenate()([second_image_hidden, LSTM_hidden_layer])
	else:
		concatenated = Concatenate()([second_image_hidden, first_noise_hidden, LSTM_hidden_layer])

	first_hidden = Dense(256, activation = "relu")(concatenated)
	first_hidden = Dropout(0.5)(first_hidden)
	output_layer = Dense(output_vector_size, activation = "softmax")(first_hidden)

	if add_noise:
		model = Model(inputs = [visible_image, visible_noise, visible_input], outputs = output_layer)
	else:
		model = Model(inputs = [visible_image, visible_input], outputs = output_layer)
	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
	return model

# This outputs a list of 50
def predict(image, n, idx2word):
	orig_lstm = np.zeros((input_words_image,))
	for i in range(input_words_image):
		if add_noise:
			predicted = generator.predict([np.array([image]), np.array([n]), np.array([orig_lstm])])
		else:
			predicted = generator.predict([np.array([image]), np.array([orig_lstm])])
		orig_lstm[i] = np.argmax(predicted[0])
	
	return " ".join([idx2word[i] for i in orig_lstm])

# amount is the number of memes
def predict_group(amount):
	for i in range(amount):
		new_sample_z = sample_z(1, input_noise_size)[0]
		print(predict(images[i], new_sample_z, idx2word))

class Histories(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		predict_group(5)
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

links, cleaned, images, vocabulary, word2idx, idx2word, vocab_embeddings, word2vec_model = initializer()

print(vocab_embeddings)

# generator = model_creator() #the inputs in order are images, noise, and lstm input
# data_generator = data_creator(cleaned, images,word2idx, 128)

# histories = Histories()

# this takes in a generator and returns the histories of the model
# generator.fit_generator(data_generator, steps_per_epoch = 3000, epochs = 15, callbacks = [histories])
