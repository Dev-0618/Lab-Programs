"""word_list = ["adventure", "journey", "quest", "mystery", "discovery", "expedition", "exploration",
"voyage"]"""

from sentence_transformers import SentenceTransformer, util
import torch, random

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sim_words(words, word_list, top_n=5):
    embeddings = model.encode([words] + word_list, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(embeddings[0], embeddings[1:]).squeeze(0)
    top = torch.topk(sims, top_n).indices.tolist()
    return [word_list[i] for i in top]

def gen_story(seed):
    words = ["adventure", "journey", "quest", "mystery", "discovery", "expedition", "exploration", "voyage"]
    sim_words = get_sim_words(seed, words, top_n=5)
    random.shuffle(sim_words)
    return (f"hello guys. a {seed} began a {sim_words[0]} and {sim_words[1]} and {sim_words[0]}")

print(gen_story("explore"))

#pip install sentence-transformers
