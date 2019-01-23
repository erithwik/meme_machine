from bs4 import BeautifulSoup
import requests
import pickle
import string
import numpy as np

'''
This takes in the list of phrases, cleans them by making them lowercase,
removing the punctuation, and removing any numbers in them 
'''
def clean_data(phrases_list):
	table = str.maketrans('', '', string.punctuation)

	cleaned_phrases_list = []
	for phrase_l in phrases_list:
		cleaned_list = []
		for i in range(len(phrase_l)):
			desc = phrase_l[i]
	        # tokenize
			desc = desc.split(" ")
	        # convert to lower case
			desc = [word.lower() for word in desc]
	        # remove punctuation from each token
			desc = [w.translate(table) for w in desc]
	        # remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
	        # store as string
			cleaned_list.append(" ".join(desc))
		cleaned_phrases_list.append(cleaned_list)

	for i in range(len(cleaned_phrases_list)):
		for j in range(len(cleaned_phrases_list[i])):
			cleaned_phrases_list[i][j] = "startseq " + cleaned_phrases_list[i][j] + " endseq"
	return cleaned_phrases_list

'''
This takes in the list of phrases and puts them in a set so that there are
no multiples (essentially a vocabulary set that used for making the word-vector)
'''
def vocabulary_creator(phrase_list):
	vocabulary = set()

	for meme in phrase_list:
		for phrase in meme:
			phrased_list = phrase.split(" ")
			for i in range(len(phrased_list)):
				vocabulary.add(phrased_list[i])

	return vocabulary

'''
This takes in the vocabulary set and returns two dictionaries, word2idx and
idx2word. The first takes in a word and returns the index of that word. The
second takes in an index and returns a word.
'''
def dictionary_creator(vocab_set):
	word2idx = {}
	idx2word = {}
	idx = 0
	for word in vocab_set:
		word2idx[word] = idx
		idx2word[idx] = word
		idx+=1

	return word2idx, idx2word

'''
This takes in a list of words and outputs the list of numbers corresponding
to the words using the word2idx dictionary.
'''
def words_to_index(list_of_words, word2idx):
	# Just to minimize the function
	return [word2idx[i] for i in list_of_words]

'''
This takes in a list of numbers and outputs a list of words that corresponds
to the indices using the idx2word dictionary.
'''
def index_to_words(list_of_indices, idx2word):
	# Just to minimize the function
	return [idx2word[i] for i in list_of_indices]

'''
Creates noise of a certain shape [m,n]
'''
def sample_Z(m, n):
	# just to minimize
    return np.random.uniform(-1, 1, size=[m, n])
