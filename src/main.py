import pandas as pd
from tqdm import tqdm

from datasets import get_articles
from modifier import long_abstract, long_title, many_articles
from recommendation import (ElasticsearchSearch, JournalVector, FastTextCNN,
                               TFIDFkNNRecommender, TFIDFRocchioRecommender)
from metric import shift_of_rank, accuracy_at_n


def run_test_for_articles(area)
    articles = get_articles(area)

    all_results = {}
    fasttext = FastTextCNN(None)
    for cases, property, modifier in ((lambda x: long_abstract(articles, 100, 199, 200),'abstract', 'long abstract'),
                                      (lambda x: long_title(articles, 100, 149, 150), 'title', 'long title 150 150'),
                                      (lambda x: long_title(articles, 100, 100, 150), 'title', 'long title 100 150'),
                                      (lambda x: many_articles(articles, 100, 200),'abstract', 'many articles 100 200'),
                                      (lambda x: many_articles(articles, 100, 300),'abstract', 'many articles 100 300'),
                                      (lambda x: many_articles(articles, 100, 400),'abstract', 'many articles 100 400'),
                                      (lambda x: many_articles(articles, 100, 500),'abstract', 'many articles 100 500'),
                                      (lambda x: many_articles(articles, 100, 600),'abstract', 'many articles 100 600'),):
        cases = cases(None)
        all_results[modifier] = {}

        es_sum100 = ElasticsearchSearch('sum', property, limit=100)
        es_average100 = ElasticsearchSearch('average', property, limit=100)
        es_sum50 = ElasticsearchSearch('sum', property, limit=50)
        es_average50 = ElasticsearchSearch('average', property, limit=50)
        jounalvec = JournalVector(property)
        knn = TFIDFkNNRecommender(property)
        rocchio = TFIDFRocchioRecommender(property)

        max_input_tok = 30
        if property == 'abstract':
            max_input_tok = 250
        fasttext.property = property
        fasttext.max_input_len = max_input_tok
        
        for recommender in (fasttext, es_sum100, es_average100, es_sum50, es_average50, jounalvec, knn, rocchio):
            results = []
            for case in tqdm(cases):
                train_data = case[case.type == 'train']
                recommender.train(train_data)
                test_data = case[case.type == 'test']
                results.append(recommender.test(test_data))

            results = pd.concat(results)
            all_results[modifier][str(recommender)] = {
                'accuracy': accuracy_at_n(results),
                'shift_of_rank': shift_of_rank(results)
            }
    print(all_results)


if __name__ == '__main__':
    run_test_for_articles('mechanics_of_materials')
    run_test_for_articles('infectious_diseases')
