
# -*- coding: utf-8 -*-

"""
 Simple example showing answering analogy questions
"""
import logging
import numpy as np
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
# from web.evaluate import evaluate_bert_analogy
from web.evaluate import get_word_vector, get_word_idx
# Configure logging
# import error from web.evaluate

def evaluate_bert_analogy(model, tokenizer, X, y, method="add", k=None, category=None, batch_size=100):

    """
    Simple method to score embedding using SimpleAnalogySolver

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings"

    X : array-like, shape (n_samples, 3)
      Analogy questions.

    y : array-like, shape (n_samples, )
      Analogy answers.

    k : int, default: None
      If not None will select k top most frequent words from embedding

    batch_size : int, default: 100
      Increase to increase memory consumption and decrease running time

    category : list, default: None
      Category of each example, if passed function returns accuracy per category
      in addition to the overall performance.
      Analogy datasets have "category" field that can be supplied here.

    Returns
    -------
    result: dict
      Results, where each key is for given category and special empty key "" stores
      summarized accuracy across categories
    """
    # if isinstance(w, dict):
    #     w = Embedding.from_dict(w)

    assert category is None or len(category) == y.shape[0], "Passed incorrect category list"
    i=0
    j=0
    w=[]
    # k=get_word_idx(i, j)
    for i in X[:, 0]:
        l = []
        for j in i.split(' '):
            l.append(get_word_vector(i, get_word_idx(i, j), tokenizer, model).numpy())
        w.append(np.mean(l, axis=0))
    # w = get_word_vector(i, j, tokenizer, model).numpy()
    solver = SimpleAnalogySolver(w=w, method=method, batch_size=batch_size, k=k)
    y_pred = solver.predict(X)

    if category is not None:
        results = OrderedDict({"all": np.mean(y_pred == y)})
        count = OrderedDict({"all": len(y_pred)})
        correct = OrderedDict({"all": np.sum(y_pred == y)})
        for cat in set(category):
            results[cat] = np.mean(y_pred[category == cat] == y[category == cat])
            count[cat] = np.sum(category == cat)
            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

        return pd.concat([pd.Series(results, name="accuracy"),
                          pd.Series(correct, name="correct"),
                          pd.Series(count, name="count")],
                         axis=1)
    else:
        return np.mean(y_pred == y)



logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch analogy dataset
data = {
    'google': fetch_google_analogy(),
    'msr': fetch_msr_analogy()
}

# for cat in (set(data.category)):
#     print(cat)

# Pick a sample of data and calculate answers
# subset = [50, 1000, 4000, 10000, 14000]
for analogy in data:
    print("ANALOGY DATASET-", analogy)
    id = 50
    w1, w2, w3 = data[analogy].X[id][0], data[analogy].X[id][1], data[analogy].X[id][2]
    print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
    print("Answer: " + data[analogy].y[id])
    # w = embeddings[e]
    tokenizer =RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
    # print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))
    a = evaluate_bert_analogy(model, tokenizer, data[analogy].X, data[analogy].y)
    print("---------------------Accuracy---------------------", e, a)

