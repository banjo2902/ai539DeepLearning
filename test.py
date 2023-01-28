import gensim.downloader
w2v = gensim.downloader.load('word2vec-google-news-300')

def analogy(a, b, c):
	global w2v
	print(a + " : " + b + " :: " + c + " : ?")
	print([(w, round(c, 3)) for w, c in w2v.most_similar(positive=[c, b], negative=[a])])

analogy('man', 'careless', 'woman')
analogy('woman', 'careless', 'man')