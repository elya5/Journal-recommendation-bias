import csv
import json

from cachalot.api import cachalot_disabled
from tqdm import tqdm

from recommender.search import recommend_journals


def generate_results(filename):
    """Generate B!SON results for articles in test file."""
    journal_index_results = {
        'title': [],
        'abstract': [],
        'all': []
    }
    checked = set()

    with open(filename) as f, cachalot_disabled():
        for line in tqdm(f):
            article = json.loads(line)
            if (article['title'] is None or len(article['title']) < 30 or
                article['abstract'] is None or len(article['abstract']) < 100):
                continue

            correct_journal = article['journal']
            if correct_journal is None or correct_journal in checked:
                continue
            else:
                checked.add(correct_journal)
            

            journalstit = recommend_journals(title=article['title'], no_nn=True,
                                             skip_article=article['idx'])
            journalsabs = recommend_journals(abstract=article['abstract'], no_nn=True,
                                             skip_article=article['idx'])
            journalsall = recommend_journals(title=article['title'],
                                            abstract=article['abstract'],
                                            skip_article=article['idx'])

            journal_index_results['title'].append((
                correct_journal, 
                get_result_index(journalstit, correct_journal)
            ))
            journal_index_results['abstract'].append((
                correct_journal, 
                get_result_index(journalsabs, correct_journal)
            ))
            journal_index_results['all'].append((
                correct_journal, 
                get_result_index(journalsall, correct_journal)
            ))

    with open('journal_index_results.json', 'w') as f:
        json.dump(journal_index_results, f)


def get_result_index(results, correct):
    """Get the place where correct journal appears in ranking."""
    for i in range(len(results)):
        if results[i]['idx'] == correct:
            return i + 1
    return None


def print_accs(name, results):
    """Print top 1, 5, 10, 15 accuracy"""
    total = len(results)

    print('Accuracy for {} (total {}): {}, {}, {}, {}'.format(
        name,
        total,
        round(len(list(filter(lambda e: e and e <= 1, results)))/total, 4),
        round(len(list(filter(lambda e: e and e <= 5, results)))/total, 4),
        round(len(list(filter(lambda e: e and e <= 10, results)))/total, 4),
        round(len(list(filter(lambda e: e and e <= 15, results)))/total, 4)
    ))


def accuracy_quartile():
    """Calculate accuracy for lower and upper quartile."""
    journals = {}
    with open('doaj_articles_per_journal.csv') as f:
        r = csv.reader(f)
        for line in r:
            journals[line[0]] = int(line[1])
    sizes = [x for x in sorted(journals.values()) if x > 30]
    lower_quartile = sizes[int(len(sizes) * 0.25)]
    upper_quartile = sizes[int(len(sizes) * 0.75)]
    print('Lower quartile', lower_quartile)
    print('Upper quartile', upper_quartile)

    lower_journals = [k for k, v in journals.items() if v <= lower_quartile]
    upper_journals = [k for k, v in journals.items() if v >= upper_quartile]

    with open('journal_index_results.json') as f:
        results = json.load(f)
    print_accs('lower quartile title',    [i for j, i in results['title'] if j in lower_journals])
    print_accs('lower quartile abstract', [i for j, i in results['abstract'] if j in lower_journals])
    print_accs('lower quartile all',      [i for j, i in results['all'] if j in lower_journals])

    print_accs('upper quartile title',    [i for j, i in results['title'] if j in upper_journals])
    print_accs('upper quartile abstract', [i for j, i in results['abstract'] if j in upper_journals])
    print_accs('upper quartile all',      [i for j, i in results['all'] if j in upper_journals])
        

if __name__ == '__main__':
    generate_results('journal_classifier/journal_classification.test.jsonl')
    accuracy_quartile()
