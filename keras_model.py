from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from web_scraper import create_data
import pickle
from keras.utils import plot_model
from utils import *

"""
So first, we need two inputs, an image with size 60*60 and a vector of 10 random 
numbers (the gaussian distribution) for additional variety. After a certain 
number of layers, the hidden layers of both become the same length. Then we 
concatenate them and send them into an LSTM which produces an output.
"""

#hyperparameters
learning_rate = 1e-3
optimizer = Adam(learning_rate)

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
def data_creator(phrases, memes, word2idx, max_length, num_photos_per_batch):
	# initializes the lists which we will later yield
	X_image, X_noise, X_captions, y_values = [], [], [], []
	# creates the counter to get the the num_photos_per_batch before resetting
	n = 0
	while True:
		# for each meme
		for index in range(0, len(memes), 15):
			# the phrases for this meme = meme_phrases
			meme_phrases = phrases[index//15]
			# for each phrase in the meme_phrases
			for meme_phrase in meme_phrases:
				# you are making a list of numbers relating to the words
				seq = [word2idx[word] for word in meme_phrase.split(' ') if word in word2idx]
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
					X_noise.append(sample_z(1,input_noise_size)[0])
					X_captions.append(input_sequences)
					y_values.append(output_word)
					# increment n for every test example
					n+=1 # <-- you can probably sub this in for len(X_image) when need be
				# if the number of test examples in the current is equal to the limit
				if n >= num_photos_per_batch:
					# this yields the batch of Xs and ys
					yield [X_image, X_noise, X_captions, y_values]
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
	second_image_hidden = Dense(128, activation = "relu")(first_image_hidden)

	visible_noise = Input((input_noise_size,))
	first_noise_hidden = Dense(128, activation = "relu")(visible_noise)

	visible_input = Input((input_words_image,))
	embeddings = Embedding(output_vector_size, embedding_dim, mask_zero=True)(visible_input)
	LSTM_hidden_layer = LSTM(128)(embeddings)

	concatenated = Add()([second_image_hidden, first_noise_hidden, LSTM_hidden_layer])

	first_hidden = Dense(256, activation = "relu")(concatenated)
	output_layer = Dense(output_vector_size, activation = "softmax")(first_hidden)

	model = Model(inputs = [visible_image, visible_noise, visible_input], outputs = output_layer)
	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
	return model

links, cleaned, images, vocabulary, word2idx, idx2word, vocab_embeddings, word2vec_model = initializer()

generator = model_creator()

output = data_creator(cleaned, images, word2idx, input_words_image, 128)
output = next(output)
X_images, X_noise, X_lstm, y_values = np.array(output[0]), np.array(output[1]), np.array(output[2]), np.array(output[3])

