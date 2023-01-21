from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
from matplotlib.style import library
import numpy as np
import nltk
# import from nltk module
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """ 
		# remove punctuation
		text = sub('\W+',' ', text)

		# run a POS tagger on the text to find out the parts-of-speech
		# and then lemmatize accordingly
		lemmatizer = WordNetLemmatizer()
		tokenizedString = []
		for word, tag in pos_tag(word_tokenize(text)):
			# nouns
			if tag.startswith("NN"):
				word = lemmatizer.lemmatize(word, pos='n')
			# verbs
			elif tag.startswith('VB'):
				word = lemmatizer.lemmatize(word, pos='v')
			# adj
			elif tag.startswith('JJ'):
				word = lemmatizer.lemmatize(word, pos='a')
				
			tokenizedString.append(word)

		return tokenizedString

	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self, corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """ 

		# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		raise UnimplementedFunctionError("You have not yet implemented build_vocab.")


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """ 

	    # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")

text = " The quick, brown fox jumped over the better but lazy dog."
print("tokenization : ", Vocabulary.tokenize(0, text))