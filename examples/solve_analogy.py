# -*- coding: utf-8 -*-

"""
 Simple example showing answering analogy questions
"""
import logging
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from web.embeddings import fetch_HDC, fetch_SG_GoogleNews, fetch_GloVe, fetch_conceptnet_numberbatch
from web.evaluate import evaluate_analogy
# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch skip-gram trained on GoogleNews corpus and clean it slightly
# w = fetch_SG_GoogleNews(lower=True, clean_words=True)
embeddings = {
    'glove': fetch_GloVe(corpus="wiki-6B", dim=300),
    'HDC': fetch_HDC(),
    'google_news': fetch_SG_GoogleNews(lower=True),
    'conceptnet_numberbatch': fetch_conceptnet_numberbatch()
}

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
    for e in embeddings:
        id = 50
        w1, w2, w3 = data[analogy].X[id][0], data[analogy].X[id][1], data[analogy].X[id][2]
        print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
        print("Answer: " + data[analogy].y[id])
        w = embeddings[e]
        # print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))
        a = evaluate_analogy(w, data[analogy].X, data[analogy].y)
        print("---------------------Accuracy---------------------", e, a)
