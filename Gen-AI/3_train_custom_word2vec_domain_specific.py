import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt_tab')

df = pd.read_csv("alldata_1_for_kaggle.csv", encoding='latin1')
medical_corpus = df['a'].to_list()

def preprocess_text(corpus):
    processed = []
    for sentence in corpus:
        tokens = word_tokenize(sentence.lower())
        # Hello World = hello world
        tokens = [word for word in tokens if word.isalpha()]
        #hello
        processed.append(tokens)
    return processed

tokenized_corpus = preprocess_text(medical_corpus)

model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, workers=3)

model.save("medical_w2v.model")
loaded_model = Word2Vec.load("medical_w2v.model")
