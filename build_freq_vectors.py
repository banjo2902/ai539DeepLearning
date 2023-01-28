from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE

import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
	"""
	    
	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

	    """ 
	try:
		# MatrixC is a matrix of co-occurence counts such that the ijth element of C denoted Cij is the number of times both wi and wj occur in a context. 
		MatrixC = np.load("CMatrix.npy")

	except:
		C = [[0]*len(vocab.word2idx)] * len(vocab.word2idx)
		for text in corpus:
			tokens = vocab.tokenize(text)

			for i in range(len(tokens)):
				for j in range(len(tokens)):
					if tokens[i] in vocab.word2idx.keys():
						if tokens[j] in vocab.word2idx.keys():
							C[vocab.word2idx[tokens[i]]][vocab.word2idx[tokens[j]]] += 1
						else:
							C[vocab.word2idx[tokens[i]]][vocab.word2idx['UNK']] += 1
					else:
						if tokens[j] in vocab.word2idx.keys():
							C[vocab.word2idx['UNK']][vocab.word2idx[tokens[j]]] += 1
						else:
							C[vocab.word2idx['UNK']][vocab.word2idx['UNK']] += 1

		MatrixC = np.array(C)
		np.save("MatrixC", MatrixC)
	
	return MatrixC
		

	# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
	# raise UnimplementedFunctionError("You have not yet implemented compute_count_matrix.")
	

###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	"""
	    
	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	    """ 
	try:
		MatrixP = np.load("MatrixP.npy")

	except:
		C = compute_cooccurrence_matrix(corpus, vocab)
		P = [[0] * C.shape[1]] * C.shape[0]
		N = len(corpus)

		for i in range(C.shape[0]):
			for j in range(C.shape[1]):
				r = (C[i, j] * N) / ((C[i, i]+0.0001) * C[j, j])

				if abs(r) <= 10**-6:
					if r > 0:
						r += 10**-6
					else:
						r -= 10**-6
				
				P[i][j] = max(0, np.log(r))
		
		MatrixP = np.array(P)
		np.save('MatrixP', MatrixP)

	return MatrixP

	# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
	# raise UnimplementedFunctionError("You have not yet implemented compute_ppmi_matrix.")


	

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]


	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)
	plt.show()


if __name__ == "__main__":
    main_freq()

