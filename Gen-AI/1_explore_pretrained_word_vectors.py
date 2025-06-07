import gensim.downloader as api
import numpy as np

word_vectors = api.load("word2vec-google-news-300")

def vector_arithmetic(w1,w2,w3):
    resultvec=word_vectors[w1]-word_vectors[w2]+word_vectors[w3]
    similarwords=word_vectors.most_similar([resultvec], topn=5)
    return similarwords

print("Result of king - man + woman")
print(vector_arithmetic("king","man","woman"))

def find_sim_words(word):
    return word_vectors.most_similar(word, topn=5)

print(" words similar to the given: \n")
print( find_sim_words("com"))


