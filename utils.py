from bs4 import BeautifulSoup
import requests
import pickle
import string

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

def vocabulary_creator(phrase_list):
	vocabulary = set()

	for meme in phrase_list:
		for phrase in meme:
			phrased_list = phrase.split(" ")
			for i in range(len(phrased_list)):
				vocabulary.add(phrased_list[i])

	return vocabulary

def dictionary_creator(vocab_set):
	word2idx = {}
	idx2word = {}
	idx = 0
	for word in vocab_set:
		word2idx[word] = idx
		idx2word[idx] = word
		idx+=1

	return word2idx, idx2word

def words_to_index(list_of_words, word2idx):
	# Just to minimize the function
	return [word2idx[i] for i in list_of_words]

def index_to_words(list_of_indices, idx2word):
	# Just to minimize the function
	return [idx2word[i] for i in list_of_indices]