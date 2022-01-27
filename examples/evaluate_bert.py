from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, RobertaTokenizer, RobertaModel

from web.datasets.similarity import fetch_MTurk, fetch_MEN, fetch_WS353, fetch_RG65, fetch_RW, fetch_SimLex999, \
    fetch_TR9856
from web.evaluate import evaluate_similarity_bert


def main():
    # Use last four layers by default
    tasks = {
        "MTurk": fetch_MTurk(),
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "Rubenstein and Goodenough": fetch_RG65(),
        "Rare Words": fetch_RW(),
        "SIMLEX999": fetch_SimLex999(),
        "TR9856": fetch_TR9856()
    }
    tokenizer =RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
    for name, data in tasks.items():
        print("Spearman correlation of scores on {} using {} {}".format(name, 'Bert',
                                                                        evaluate_similarity_bert(model, tokenizer,
                                                                                                 data.X, data.y)))


if __name__ == '__main__':
    main()
