import cohere, nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

co = cohere.Client("xSfqQtfn5Yw4lSguFJnzVNDhdexYMTzQwTLTVtsT")

def enhance_prompt(prompt):
    words = word_tokenize(prompt)
    tags = pos_tag(words)
    targets = [w for w, t in tags if t in ["NN", "NNS", "JJ"]]
    if not targets: return prompt
    embeddings = np.array(co.embed(texts=targets, model="embed-english-v3.0", input_type="search_document").embeddings)
    enriched = []
    for w in words:
        if w in targets:
            i = targets.index(w)
            sim = cosine_similarity([embeddings[i]], embeddings)[0]
            enriched.append(targets[np.argsort(sim)[-2]])
        else:
            enriched.append(w)
    return " ".join(enriched)

original = "Describe the impact of artificial intelligence on healthcare."
enriched = enhance_prompt(original)
print("Original Prompt:", original)
print("Enriched Prompt:", enriched)
