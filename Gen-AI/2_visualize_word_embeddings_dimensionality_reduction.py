"""Use dimensionality reduction (e.g., PCA or t-SNE) to visualize word
embeddings for Q 1. Select 10 words from a specific domain (e.g., sports,
technology) and visualize their embeddings. Analyze clusters and
relationships. Generate contextually rich outputs using embeddings. Write
a program to generate 5 semantically similar words for a given input.
"""


import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

word_vectors = api.load("word2vec-google-news-300")

words = ["computer", "cybersecurity", "machine"]
vectors = np.array([word_vectors[word] for word in words])

def plot_embeddings(vectors, words, method="PCA"):
    reducer = PCA(n_components=2) if method == "PCA" else TNSE(n_components=2, perplexity=5, random_state=42)
    reduced = reducer.fit_transformer(vectors)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1])
    for i, word in enumerate(words):
        plt.annotate(words, (reduced[i,0], reduced[i,1]), fontsize=12)

    plt.title(f"word em {method}")
    plt.show()

#pip install scikit-learn 
#pip install matplotlib.pyplot
#pip install gensim
#pip install numpy
