import numpy as np
import nltk
import gensim
from gensim import corpora, models, similarities
import pickle
from utils import *
nltk.download("punktm")

'''
This function takes in a corpus of data (a list of strings where each string
represents a full sentence) and it makes a word embedding based on the Word2Vec
algorithm. It also takes in an embedding_length which is the length of the 
vector corresponding to each word. The function also saves the model if the 
save_model parameter is set to true (it is originally set to false). The function
outputs a model and you can use functions with this model to access some words.
Ex:
model = create_embeddings(corpus, 200)
print(model["you"]) would output the 200 dim vector that corresponds to this word
print(model.most_similar("you")) would output some words that the algorithm 
considers similar to the word
'''
def create_embeddings(corpus, embedding_length, save_model = False):
	tok_corp = [nltk.word_tokenize(sent) for sent in corpus]
	model = gensim.models.Word2Vec(tok_corp, min_count = 1, size = embedding_length)
	if(save_model):
		with open("data_storage/word_model.pickle", "wb") as f:
			pickle.dump(model, f)
	return model

with open("data_storage/word_model.pickle", "rb") as f:
	model = pickle.load(f)
