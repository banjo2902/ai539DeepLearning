from collections import Counter, defaultdict
from email.policy import default 
import re
from re import sub, compile
import matplotlib.pyplot as plt
from matplotlib.style import library
import numpy as np
from datasets import load_dataset
# import from nltk module
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# dataset = load_dataset("ag_news")
# dataset_text =  [r['text'] for r in dataset['train']]
# print("dataset length: \n", dataset['train'].num_rows)
# print("dataset text: \n", dataset_text[:3])
# # print("features: \n", dataset['train'].features.get())
# # print("num_rows : ", dataset.num_rows)

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):
		print("start!\n")
		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]

	def mostFreqTokens(self, freq, k):
		sortedFreq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
		l = 1
		# print("sorted_freq: ", sortedFreq)
		for key, value in sortedFreq:
			k -= value
			if k <= 0: break
			else: l += 1
		
		return [t for t,f in sortedFreq[:l]]

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
		# text = sub('\W+',' ', text)

		# run a POS tagger on the text to find out the parts-of-speech
		# and then lemmatize accordingly
		# lemmatizer = WordNetLemmatizer()
		# tokenizedString = lemmatizer.lemmatize(text)
		# for word, tag in pos_tag(word_tokenize(text)):
		# 	# nouns
		# 	if tag.startswith("NN"):
		# 		word = lemmatizer.lemmatize(word, pos='n')
		# 	# verbs
		# 	elif tag.startswith('VB'):
		# 		word = lemmatizer.lemmatize(word, pos='v')
		# 	# adj
		# 	elif tag.startswith('JJ'):
		# 		word = lemmatizer.lemmatize(word, pos='a')
				
		# 	tokenizedString.append(word)

		# remove space and punctuation
		# text = list(filter(None, re.split('[ :;,.!?()-\/]', text)))

		lemmatizer = RegexpTokenizer(r'\w+')
		tokenizedStrings = lemmatizer.tokenize(text)
		# print("tokenizedStrings: \n", tokenizedStrings)
		return tokenizedStrings

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
		# freq
		freq = defaultdict(int)
		for text in corpus:
			tokens = self.tokenize(text)
			for token in tokens:
				freq[token] += 1

		# get most common words(list)
		sortedFreq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
		k = sum(freq.values()) * 0.9
		l = 1
		Threshold = -1

		for key, value in sortedFreq:
			k -= value
			if k <= 0: 
				Threshold = value
				break
			else: l += 1
		print("Threshold: ", Threshold)
		print("l: ", l)
		commonTokens = [t for t,f in sortedFreq[:l]]
		# print("commonWords : ", commonWords)

		# word2idx
		word2idx = {'UNK': 0}
		numericalIdx = 1
		self.totalTokensNum = 1
		for text in corpus:
			tokens = self.tokenize(text)
			self.totalTokensNum += 1
			for token in tokens:
				if token in commonTokens:
					if token not in word2idx:
						word2idx[token] = numericalIdx
						numericalIdx += 1

		# inverse map = {v: k for k, v in my_map.items()}
		idx2word = {value: key for key, value in word2idx.items()}
		
		return word2idx, idx2word, freq

	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details


	    """ 
		# sortedFreqlist = sorted(self.freq.values(), reverse=True)
		sortedFreqlist = [v for v in self.freq.values()]
		sortedFreqlist.sort(reverse=True)
		cumuCoverage = 0
		fracs = []

		plt.figure(figsize=(10, 8))
		plt.plot([i for i in range(1, len(sortedFreqlist)+1)], sortedFreqlist)
		plt.title("Token Frequency Distribution")
		plt.xlabel("Token ID")
		plt.ylabel("Frequency")
		plt.show()

		for i in sortedFreqlist:
			cumuCoverage += i
			fracs.append(cumuCoverage / self.totalTokensNum)
		
		plt.figure(figsize=(10, 8))
		plt.plot([i for i in range(1, len(fracs)+1)], fracs)
		plt.title("Cumulative Fraction Covered")
		plt.xlabel("Token ID (sorted by frequency)")
		plt.ylabel("Frequency Token Occureneces Covered")
		plt.show()
		print()

		# print(sortedFreqlist[20001])
		# print(fracs[20001])

	    # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		# raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")