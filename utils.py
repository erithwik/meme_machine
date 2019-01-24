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
def sample_z(m, n):
	# just to minimize
	return np.random.uniform(-1, 1, size=[m, n])

'''
This takes the word2idx and the word2vec model and creates a numpy array 
of size (num_words, num_embeddings) which holds the information about
each word.
'''
def make_numpy_embedding_list(num_words, num_embeddings, word2idx, model):
	vocab_set = np.zeros((num_words, num_embeddings))

	for word, i in word2idx.items():
		try:
			vector = model[word]
		except KeyError as e:
			vector = None

		if vector is not None:
			vocab_set[i] = vector

	return vocab_set


'''
This pads the curr_list with zeroes to reach max_length
'''
def pad_sequences(curr_list, max_length):
	# just to minimize
	return curr_list + [0] * (max_length - len(curr_list))

'''
This returns a list where one index is 1 which corresponds to the y
value
'''
def categorize_variable(variable, vocab_size):
	ret_list = [0] * vocab_size
	ret_list[variable] = 1
	return ret_list
