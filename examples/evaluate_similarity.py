# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging

from six import iteritems

from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, \
    fetch_TR9856
from web.embeddings import fetch_HDC, fetch_SG_GoogleNews, fetch_GloVe, fetch_conceptnet_numberbatch
from web.evaluate import evaluate_similarity

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch embeddings (warning: it might take few minutes)
embeddings = {
    'glove': fetch_GloVe(corpus="wiki-6B", dim=300),
    'HDC': fetch_HDC(),
    'google_news': fetch_SG_GoogleNews(lower=True),
    'conceptnet_numberbatch': fetch_conceptnet_numberbatch()
}

# Define tasks
tasks = {
    "MTurk": fetch_MTurk(),
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "Rubenstein and Goodenough": fetch_RG65(),
    "Rare Words": fetch_RW(),
    "SIMLEX999": fetch_SimLex999(),
    "TR9856": fetch_TR9856()
}

# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1],
                                                                                    data.y[0]))

# Calculate results using helper function
for e_name, e in embeddings.items():
    for name, data in iteritems(tasks):
        print("Spearman correlation of scores on {} using {} {}".format(name, e_name,
                                                                        evaluate_similarity(e, data.X, data.y)))
