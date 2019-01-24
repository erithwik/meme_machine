from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from web_scraper import create_data
import pickle
from keras.utils import plot_model

"""
So first, we need two inputs, an image with size 60*60 and a vector of 10 random 
numbers (the gaussian distribution) for additional variety. After a certain 
number of layers, the hidden layers of both become the same length. Then we 
concatenate them and send them into an LSTM which produces an output.
"""

#hyperparameters
learning_rate = 1e-3
optimizer = Adam(learning_rate)

input_image_size = 3600
input_noise_size = 100
input_words_image = 50 # the max was 24 words in the test data
output_vector_size = 821
embedding_dim = 200
# creating the generator
def model_creator():
	visible_image = Input((input_image_size,))
	first_image_hidden = Dense(256, activation = "relu")(visible_image)
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

created_model = model_creator()
print(created_model.summary())
