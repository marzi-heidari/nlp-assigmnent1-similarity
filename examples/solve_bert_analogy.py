
# -*- coding: utf-8 -*-

"""
 Simple example showing answering analogy questions
"""
import logging
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from web.evaluate import evaluate_bert_analogy
# Configure logging
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