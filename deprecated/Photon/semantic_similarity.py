from pprint import pprint

# from tabulate import tabulate
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

# warnings.filterwarnings("ignore")
from PhotonicsAI.Photon import utils

# if not nltk.data.find('D:\\vahid\\PhotonEnv\\nltk_data'):
#     nltk.download('punkt', download_dir='D:\\vahid\\PhotonEnv\\nltk_data')


def st(prompt, list_of_items):
    model = SentenceTransformer("paraphrase-albert-small-v2")

    items_embeddings = model.encode(list_of_items, convert_to_tensor=True)

    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    cosine_scores = util.cos_sim(prompt_embedding, items_embeddings)

    indices = list(range(len(list_of_items)))
    scores = [score.item() for score in cosine_scores[0]]

    indices = (np.argsort(scores)[::-1]).tolist()
    return indices, scores


def dragon(query, contexts):
    tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
    query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
    context_encoder = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")

    # Apply tokenizer
    query_input = tokenizer(query, return_tensors="pt")
    ctx_input = tokenizer(contexts, padding=True, truncation=False, return_tensors="pt")
    # Compute embeddings: take the last-layer hidden state of the [CLS] token
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
    # Compute similarity scores using dot product

    s = []
    for i in range(len(contexts)):
        score = query_emb @ ctx_emb[i]
        # print(i, score)
        s.append(score.item())

    indices = (np.argsort(s)[::-1]).tolist()
    return indices, s


def bm25(query, contexts):
    # Tokenizing documents
    tokenized_docs = [word_tokenize(doc.lower()) for doc in contexts]

    # Create a BM25 Object
    bm25 = BM25Okapi(tokenized_docs)

    # Tokenize query
    tokenized_query = word_tokenize(query.lower())

    # Get scores
    doc_scores = bm25.get_scores(tokenized_query)

    # get the top N documents
    bm25.get_top_n(tokenized_query, contexts, n=len(contexts))

    indices = (np.argsort(doc_scores)[::-1]).tolist()

    # print(doc_scores)
    # print('=======')
    # print(top_docs)
    return indices, doc_scores


def combine_scores(score_lists):
    score_array = np.array(score_lists)

    # there are more sophisticated ways of aggregating the scores, but for now:
    # geometric_mean. Adding small number to avoid log(0)
    combined_scores = np.exp(np.mean(np.log(score_array + 1e-10), axis=0))

    indices = (np.argsort(combined_scores)[::-1]).tolist()
    return indices, combined_scores


if __name__ == "__main__":
    db_docs = utils.search_directory_for_docstrings()
    list_of_desc = [i["Description"] for i in db_docs]

    query = "an mzi for c-band"
    # st('MZI', list_of_desc)
    i, s = dragon(query, list_of_desc)
    print(i, s)

    ii, ss = bm25(query, list_of_desc)
    print(ii, ss)

    final_i, final_s = combine_scores([s, ss])
    print(final_i, final_s)

    sorted_names = [db_docs[i]["Name"] for i in final_i]
    pprint(sorted_names, width=200)
