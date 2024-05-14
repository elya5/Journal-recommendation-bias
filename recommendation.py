import math
import os
import random
import re
import shutil
import time
import zipfile

from elasticsearch import helpers
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Index
import fasttext
import nltk
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import train_test_split
import spacy
from textblob import TextBlob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


RANDOM_STATE=42
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
pl.seed_everything(seed=RANDOM_STATE)
torch.backends.cudnn.benchmark = False


class ElasticsearchSearch:
    def __init__(self, aggregate, property, limit=150):
        self.limit = limit
        self.aggregate = aggregate
        if aggregate not in ('sum', 'average'):
            raise ValueError()
        self.property = property
        self.index_name = 'test_index'
        self.es = Elasticsearch(
            [{'host': 'localhost', 'port': 9200}],
            timeout=60
        )

    def __str__(self):
        return f'Elasticsearch <{self.aggregate},{self.property},{self.limit}>'

    def train(self, df):
        assert len(df[df.type=='test']) == 0
        articles = df[[self.property, 'journal_id']].to_dict('records')
        self.delete_index()
        self.create_index()
        errors = 0
        for ok, action in helpers.streaming_bulk(
                self.es,
                articles,
                index=self.index_name):
            if not ok:
                print('\nError:', action)
                errors += 1
        if errors:
            print('Errors during indexing')
        while Search(using=self.es, index=self.index_name).count() == 0:
            time.sleep(5)

    def delete_index(self):
        index = Index(self.index_name, using=self.es)
        if index.exists():
            Index(self.index_name, using=self.es).delete()

    def create_index(self):
        index = Index(self.index_name, using=self.es)
        index.create()

    def test(self, df):
        assert len(df[df.type=='train']) == 0
        results = []
        for _, article in df.iterrows():
            if self.property == 'title':
                hits = (Search(using=self.es, index=self.index_name)
                        .query('match', title=article.title)
                        .params(size=self.limit, request_timeout=40)
                        .execute())
            elif self.property == 'abstract':
                hits = (Search(using=self.es, index=self.index_name)
                        .query('match', abstract=article.abstract)
                        .params(size=self.limit, request_timeout=40)
                        .execute())

            df_results = pd.DataFrame([
                {'journal_id': hit['journal_id'], 'score': hit.meta.score}
                for hit in hits
            ])
            if df_results.size == 0:
                print('warning: no results')
                df_results = pd.DataFrame([{'journal_id': '-', 'score': 0}])

            if self.aggregate == 'sum':
                df_results = df_results.groupby('journal_id')['score'].sum().reset_index()
            elif self.aggregate == 'average':
                df_results = df_results.groupby('journal_id')['score'].mean().reset_index()

            df_results['true_journal'] = article.journal_id
            df_results['modified_journal'] = article.modified_journal_id
            df_results['method'] = f'Elasticsearch<{self.aggregate},{self.property}>'
            df_results['doi'] = article.doi

            results.append(df_results)
        return pd.concat(results)



class JournalVector:
    def __init__(self, property):
        self.nlp = spacy.load(
            "en_core_web_md",
            disable=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]
        )
        self.property = property
        self.nlp.max_length = 10_000_000

    def __str__(self):
        return f'JournalVector <{self.property}>'

    def train(self, df):
        assert len(df[df.type=='test']) == 0
        self.trained_df = df.groupby('journal_id').apply(lambda g: ' '.join(g[self.property])).to_frame(name='abstracts').reset_index()

        self.trained_df['embedding'] = self.trained_df.abstracts.apply(lambda a: self.nlp(a))

    def test(self, df):
        assert len(df[df.type=='train']) == 0

        results = []
        for _, article in df.iterrows():
            doc = self.nlp(article[self.property])
            df_results = self.trained_df.copy()
            df_results['score'] = df_results.embedding.apply(lambda emb: emb.similarity(doc))
            df_results['true_journal'] = article.journal_id
            df_results['modified_journal'] = article.modified_journal_id
            df_results['method'] = f'JournalVector'
            df_results['doi'] = article.doi

            results.append(df_results)
        return pd.concat(results)



class TFIDFRocchioRecommender:
    def __init__(self, property):
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',
                                          min_df=3, max_df=570/583)
        self.clf = NearestCentroid(metric='cosine')
        self.regex1 = re.compile(r'\d+')
        self.property = property

    def __str__(self):
        return f'TFIDFRocchioRecommender <{self.property}>'

    def preprocess(self, text):
        ps = PorterStemmer()
        text = text.lower()
        text = self.regex1.sub("", text)
        tokenized = TextBlob(text).words
        tokenized = [
            ''.join([char
                    for char in token
                    if char not in "!@#$%^&*()[]{};:,./<>?\\|`~=_+'"])
            for token in tokenized
        ]
        return ' '.join([ps.stem(w) for w in tokenized])

    def train(self, df):
        assert len(df[df.type=='test']) == 0

        df_mod = df.copy()
        df_mod['preprocessed'] = df_mod[self.property].apply(lambda t: self.preprocess(t))
        train_tfidf = self.vectorizer.fit_transform(df_mod.preprocessed)
        self.clf.fit(train_tfidf, df_mod.journal_id)

    def test(self, df):
        assert len(df[df.type=='train']) == 0
        df_mod = df.copy()
        df_mod['preprocessed'] = df_mod[self.property].apply(lambda t: self.preprocess(t))

        results = []
        for _, article in df_mod.iterrows():
            distances = pairwise_distances(
                self.clf.centroids_,
                self.vectorizer.transform([article[self.property]]),
                metric=self.clf.metric
            )
            df_results = pd.DataFrame({'journal_id': self.clf.classes_, 'score': 1-distances[:,0]})
            df_results['true_journal'] = article.journal_id
            df_results['modified_journal'] = article.modified_journal_id
            df_results['method'] = 'TFIDFRocchioRecommender'
            df_results['doi'] = article.doi

            results.append(df_results)

        return pd.concat(results)


class TFIDFkNNRecommender(TFIDFRocchioRecommender):
    def __init__(self, property, neighbors=50):
        super().__init__(property)
        self.n_neighbors = neighbors
        self.clf = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
        self.property = property

    def __str__(self):
        return f'TFIDFkNNRecommender <{self.property},{self.n_neighbors}>'

    def test(self, df):
        assert len(df[df.type=='train']) == 0
        df_mod = df.copy()
        df_mod['preprocessed'] = df_mod[self.property].apply(lambda t: self.preprocess(t))

        results = []
        for _, article in df_mod.iterrows():
            probs = self.clf.predict_proba(
                self.vectorizer.transform([article[self.property]])
            )[0]
            df_results = pd.DataFrame({'journal_id': self.clf.classes_, 'score': probs})
            df_results['true_journal'] = article.journal_id
            df_results['modified_journal'] = article.modified_journal_id
            df_results['method'] = 'TFIDFkNNRecommender'
            df_results['doi'] = article.doi

            results.append(df_results)

        return pd.concat(results)





############ NN


class FastTextCNN:
    def __init__(self, property, batch_size=50, epochs=5, max_input_len=150):
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_input_len = max_input_len
        self.fasttext_model = self.load_fasttext_wiki_en()
        self.property = property

    def __str__(self):
        return f'FastTextCNN <{self.batch_size},{self.epochs},{self.max_input_len}>'

    def load_fasttext_wiki_en(self):
        file_path = 'data/wiki.en.zip'
        if not os.path.exists('data/wiki_en'):
            url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip'
            r = requests.get(url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                os.fsync(f.fileno())

            with zipfile.ZipFile(file_path, 'r') as f:
                f.extractall('data/wiki_en')
            os.remove('data/wiki_en/wiki.en.vec')
            os.remove(file_path)
        return fasttext.load_model('data/wiki_en/wiki.en.bin')

    def train(self, df):
        assert len(df[df.type=='test']) == 0
        self.classlabels = {journal: i for i, journal in enumerate(df.journal_id.unique())}
        num_classes = len(self.classlabels)

        df_train, df_val = train_test_split(
            df, 
            test_size=min(200, math.ceil(df.shape[0]*0.2)), 
            random_state=RANDOM_STATE, stratify=df.journal_id
        )
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        data_module = TextClassificationDataModule(
            df_train,
            df_val,
            None,
            self.batch_size,
            self.fasttext_model,
            self.classlabels,
            self.max_input_len,
            self.property,
        )
        self.model = TextClassifierLightning(num_classes, self.max_input_len)

        self.trainer = pl.Trainer(max_epochs=self.epochs) #, gpus=1)
        self.trainer.fit(self.model, data_module)

    def test(self, df):
        assert len(df[df.type=='train']) == 0

        data_module = TextClassificationDataModule(
            None,
            None,
            df.copy(),
            self.batch_size,
            self.fasttext_model,
            self.classlabels,
            self.max_input_len,
            self.property,
        )
        self.trainer.test(self.model, datamodule=data_module)
        self.model.eval()

        results = []
        inverse_classlabels = {v:k for k,v in self.classlabels.items()}
        test_dataset = TextClassificationDataset(df, self.fasttext_model, self.classlabels, self.max_input_len, self.property)
        for i in range(df.shape[0]):
            row = test_dataset.get_datarow(i)
            result = self.model(test_dataset[i][0].unsqueeze(0))[0]
            df_results = pd.DataFrame([
                {'journal_id': inverse_classlabels[i], 'score': prob.item()}
                for i, prob in enumerate(result)
            ])
            df_results['true_journal'] = row.journal_id
            df_results['modified_journal'] = row.modified_journal_id
            df_results['method'] = 'FastTextCNN'
            df_results['doi'] = row.doi
            results.append(df_results)

        # remove temp files
        shutil.rmtree('lightning_logs')

        return pd.concat(results)


class TextClassifierLightning(pl.LightningModule):
    def __init__(self, num_classes, max_input_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=450, kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(int(450*((max_input_len-2)/4)), 250)
        self.fc2 = nn.Linear(250, num_classes)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x, labels=None):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        if labels is not None:
            return self.criterion(output, labels)
        else:
            return self.softmax(output)


    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        acc = self.accuracy(preds, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        acc = self.accuracy(preds, y)
        self.log('text_acc', acc, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.5e-3)
        return optimizer


class TextClassificationDataset(Dataset):
    def __init__(self, df_articles, fasttext_model, classlabels, max_input_len, property):
        stop_words = list(stopwords.words('english'))

        df_articles = df_articles.sample(frac=1, random_state=RANDOM_STATE).reset_index()
        if property == 'title':
            df_articles['text'] = df_articles.title.fillna('').str.lower()
        elif property == 'abstract':
            df_articles['text'] = df_articles.abstract.fillna('').str.lower()
        else:
            df_articles['text'] = df_articles.title.fillna('').str.lower() + ' ' + df_articles.abstract.fillna('').str.lower()

        def sanitize_text(text):
            text = ''.join([letter
                            for letter in text
                            if letter.isalpha() or letter == ' '])
            text = [word for word in text.split() if word not in stop_words]
            return ' '.join(text)

        df_articles.text = df_articles.text.apply(sanitize_text)
        df_articles['label'] = df_articles.journal_id.apply(classlabels.get)

        self.df_articles = df_articles

        self.fasttext_model = fasttext_model
        self.max_input_len = max_input_len

    def __len__(self):
        return self.df_articles.shape[0]

    def get_datarow(self, idx):
        return self.df_articles.iloc[idx]

    def __getitem__(self, idx):
        text = self.df_articles.iloc[idx].text
        label = self.df_articles.iloc[idx].label

        tokens = fasttext.tokenize(text)
        embeddings = [
            self.fasttext_model.get_word_vector(token)
            for token in tokens
        ][:self.max_input_len]
        embeddings = np.array(embeddings)
        if embeddings.shape[0] < self.max_input_len:
            embeddings = np.concatenate((
                embeddings,
                np.zeros((self.max_input_len - embeddings.shape[0], 300))
            ))

        return torch.FloatTensor(embeddings.T), torch.tensor(label, dtype=torch.long)


class TextClassificationDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size, fasttext_model, classlabels, max_input_len, property):
        super(TextClassificationDataModule, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.fasttext_model = fasttext_model
        self.classlabels = classlabels
        self.max_input_len = max_input_len
        self.property = property

    def setup(self, stage=None):
        if self.train_data is not None:
            self.train_dataset = TextClassificationDataset(self.train_data, self.fasttext_model, self.classlabels, self.max_input_len, self.property)
        if self.val_data is not None:
            self.val_dataset = TextClassificationDataset(self.val_data, self.fasttext_model, self.classlabels, self.max_input_len, self.property)
        if self.test_data is not None:
            self.test_dataset = TextClassificationDataset(self.test_data, self.fasttext_model, self.classlabels, self.max_input_len, self.property)

    def train_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            worker_init_fn=seed_worker,
            generator=g,
        )


##### tests


def test_fasttextcnn():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'title': 'Plasma interleukin-41 serves as a potential diagnostic biomarker for Kawasaki disease.'},
        {'type': 'train', 'journal_id': 'x', 'title': 'Single-cell dissection of the human motor and prefrontal cortices in ALS and FTLD.'},
        {'type': 'train', 'journal_id': 'x', 'title': 'The effects of pregnancy, its progression, and its cessation on human (maternal) biological aging.'},
        {'type': 'train', 'journal_id': 'x', 'title': 'A gut-derived hormone regulates cholesterol metabolism.'},
        {'type': 'train', 'journal_id': 'x', 'title': 'Ribociclib plus Endocrine Therapy in Early Breast Cancer'},

        {'type': 'train', 'journal_id': 'y', 'title': 'On a B-field transform of generalized complex structures over complex tori'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Random vortex dynamics and Monte-Carlo simulations for wall-bounded viscous flows'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Comparing functional countability and exponential separability'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Sparse additive function decompositions facing basis transforms'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Non-existence of Ulrich modules over Cohen-Macaulay local rings'},

        {'type': 'test', 'journal_id': 'x', 'doi': 'doix', 'modified_journal_id': 'a', 'title': ' A distinct Fusobacterium nucleatum clade dominates the colorectal cancer niche.'},
        {'type': 'test', 'journal_id': 'y', 'doi': 'doiy', 'modified_journal_id': 'b', 'title': ' Highly connected graphs have highly connected spanning bipartite subgraphs'},
    ])

    df_train = df[df.type == 'train']
    df_test = df[df.type == 'test']

    cnn = FastTextCNN(max_input_len=8, property='title')
    cnn.train(df_train)
    results = cnn.test(df_test)
    results = results.reset_index(drop=True)

    assert results.shape[0] <= 4
    assert results.loc[results[(results.doi == 'doix')].score.idxmax()].journal_id == 'x'
    assert results.loc[results[(results.doi == 'doiy')].score.idxmax()].journal_id == 'y'
    assert len(results.method.iloc[0]) > 5
    assert len(results.doi.unique()) == 2
    assert results[results.doi == 'doix'].true_journal.iloc[0] == 'x'
    assert results[results.doi == 'doiy'].true_journal.iloc[0] == 'y'
    assert results[results.doi == 'doiy'].modified_journal.iloc[0] == 'b'
    assert results[results.doi == 'doix'].modified_journal.iloc[0] == 'a'


def test_fasttextcnn_same_abstract_title():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'abstract': 'x', 'title': 'Plasma interleukin-41 serves as a potential diagnostic biomarker for Kawasaki disease.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'x', 'title': 'Single-cell dissection of the human motor and prefrontal cortices in ALS and FTLD.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'x', 'title': 'The effects of pregnancy, its progression, and its cessation on human (maternal) biological aging.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'x', 'title': 'A gut-derived hormone regulates cholesterol metabolism.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'x', 'title': 'Ribociclib plus Endocrine Therapy in Early Breast Cancer'},

        {'type': 'train', 'journal_id': 'y', 'abstract': 'x', 'title': 'On a B-field transform of generalized complex structures over complex tori'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'x', 'title': 'Random vortex dynamics and Monte-Carlo simulations for wall-bounded viscous flows'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'x', 'title': 'Comparing functional countability and exponential separability'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'x', 'title': 'Sparse additive function decompositions facing basis transforms'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'x', 'title': 'Non-existence of Ulrich modules over Cohen-Macaulay local rings'},

        {'type': 'test', 'journal_id': 'x', 'doi': 'doix', 'modified_journal_id': 'a', 'abstract': 'x', 'title': ' A distinct Fusobacterium nucleatum clade dominates the colorectal cancer niche.'},
        {'type': 'test', 'journal_id': 'y', 'doi': 'doiy', 'modified_journal_id': 'b', 'abstract': 'x', 'title': ' Highly connected graphs have highly connected spanning bipartite subgraphs'},
    ])

    df_train = df[df.type == 'train']
    df_test = df[df.type == 'test']

    cnn = FastTextCNN(max_input_len=8, property='title')
    cnn.train(df_train)
    results_title = cnn.test(df_test)
    results_title = results_title.reset_index(drop=True)

    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'abstract': 'Plasma interleukin-41 serves as a potential diagnostic biomarker for Kawasaki disease.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'Single-cell dissection of the human motor and prefrontal cortices in ALS and FTLD.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'The effects of pregnancy, its progression, and its cessation on human (maternal) biological aging.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'A gut-derived hormone regulates cholesterol metabolism.'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'Ribociclib plus Endocrine Therapy in Early Breast Cancer'},

        {'type': 'train', 'journal_id': 'y', 'abstract': 'On a B-field transform of generalized complex structures over complex tori'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Random vortex dynamics and Monte-Carlo simulations for wall-bounded viscous flows'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Comparing functional countability and exponential separability'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Sparse additive function decompositions facing basis transforms'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Non-existence of Ulrich modules over Cohen-Macaulay local rings'},

        {'type': 'test', 'journal_id': 'x', 'doi': 'doix', 'modified_journal_id': 'a', 'abstract': 'A distinct Fusobacterium nucleatum clade dominates the colorectal cancer niche.'},
        {'type': 'test', 'journal_id': 'y', 'doi': 'doiy', 'modified_journal_id': 'b', 'abstract': 'Highly connected graphs have highly connected spanning bipartite subgraphs'},
    ])
    df_train = df[df.type == 'train']
    df_test = df[df.type == 'test']

    del cnn
    cnn = FastTextCNN(max_input_len=8, property='abstract')

    cnn.train(df_train)
    results_abstract = cnn.test(df_test)
    results_abstract = results_abstract.reset_index(drop=True)

    assert results_title.loc[results_title[(results_title.doi == 'doix')].score.idxmax()].journal_id  == results_abstract.loc[results_abstract[(results_abstract.doi == 'doix')].score.idxmax()].journal_id
    assert results_title.loc[results_title[(results_title.doi == 'doiy')].score.idxmax()].journal_id  == results_abstract.loc[results_abstract[(results_abstract.doi == 'doiy')].score.idxmax()].journal_id


def test_systems_abstract():
    es_mean = ElasticsearchSearch('average', 'abstract')
    es_sum = ElasticsearchSearch('sum', 'abstract')
    rocchio = TFIDFRocchioRecommender('abstract')
    knn = TFIDFkNNRecommender('abstract', neighbors=3)
    jv = JournalVector('abstract')

    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'abstract': 'hospital and medicine'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'AI in medicine'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'hospitals use AI'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'bacteria eat all the microspes. Can AI help'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'bacteria and micropes for medicine'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'do micropes live in hospitals'},

        {'type': 'train', 'journal_id': 'y', 'abstract': 'motor vehicles'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'fire in the motor'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'why my car does not drive'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'car engines in vehicles'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'are trucks really vehicles'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'motor in car necessary?'},

        {'type': 'test', 'journal_id': 'y', 'abstract': 'vehicles with motor is car', 'doi': 'doiy', 'modified_journal_id': 'b'},
        {'type': 'test', 'journal_id': 'x', 'abstract': 'hospitals use bacteria as medicine', 'doi': 'doix', 'modified_journal_id': 'a'},
    ])
    df_train = df[df.type == 'train']
    df_test = df[df.type == 'test']

    for system in (es_mean, es_sum, rocchio, knn, jv):
        system.train(df_train)
        results = system.test(df_test)
        results = results.reset_index(drop=True)

        assert results.shape[0] <= 4
        assert results.loc[results[(results.doi == 'doix')].score.idxmax()].journal_id == 'x'
        assert results.loc[results[(results.doi == 'doiy')].score.idxmax()].journal_id == 'y'
        assert len(results.method.iloc[0]) > 5
        assert len(results.doi.unique()) == 2
        assert results[results.doi == 'doix'].true_journal.iloc[0] == 'x'
        assert results[results.doi == 'doiy'].true_journal.iloc[0] == 'y'
        assert results[results.doi == 'doiy'].modified_journal.iloc[0] == 'b'
        assert results[results.doi == 'doix'].modified_journal.iloc[0] == 'a'


def test_systems_title():
    es_mean = ElasticsearchSearch('average', 'title')
    es_sum = ElasticsearchSearch('sum', 'title')
    rocchio = TFIDFRocchioRecommender('title')
    knn = TFIDFkNNRecommender('title', neighbors=3)
    jv = JournalVector('title')

    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'title': 'hospital and medicine'},
        {'type': 'train', 'journal_id': 'x', 'title': 'AI in medicine'},
        {'type': 'train', 'journal_id': 'x', 'title': 'hospitals use AI'},
        {'type': 'train', 'journal_id': 'x', 'title': 'bacteria eat all the microspes. Can AI help'},
        {'type': 'train', 'journal_id': 'x', 'title': 'bacteria and micropes for medicine'},
        {'type': 'train', 'journal_id': 'x', 'title': 'do micropes live in hospitals'},

        {'type': 'train', 'journal_id': 'y', 'title': 'motor vehicles'},
        {'type': 'train', 'journal_id': 'y', 'title': 'fire in the motor'},
        {'type': 'train', 'journal_id': 'y', 'title': 'why my car does not drive'},
        {'type': 'train', 'journal_id': 'y', 'title': 'car engines in vehicles'},
        {'type': 'train', 'journal_id': 'y', 'title': 'are trucks really vehicles'},
        {'type': 'train', 'journal_id': 'y', 'title': 'motor in car necessary?'},

        {'type': 'test', 'journal_id': 'y', 'title': 'vehicles with motor is car', 'doi': 'doiy', 'modified_journal_id': 'b'},
        {'type': 'test', 'journal_id': 'x', 'title': 'hospitals use bacteria as medicine', 'doi': 'doix', 'modified_journal_id': 'a'},
    ])

    df_train = df[df.type == 'train']
    df_test = df[df.type == 'test']

    for system in (es_mean, es_sum, rocchio, knn, jv):
        system.train(df_train)
        results = system.test(df_test)
        results = results.reset_index(drop=True)

        assert results.shape[0] <= 4
        assert results.loc[results[(results.doi == 'doix')].score.idxmax()].journal_id == 'x'
        assert results.loc[results[(results.doi == 'doiy')].score.idxmax()].journal_id == 'y'
        assert len(results.method.iloc[0]) > 5
        assert len(results.doi.unique()) == 2
        assert results[results.doi == 'doix'].true_journal.iloc[0] == 'x'
        assert results[results.doi == 'doiy'].true_journal.iloc[0] == 'y'
        assert results[results.doi == 'doiy'].modified_journal.iloc[0] == 'b'
        assert results[results.doi == 'doix'].modified_journal.iloc[0] == 'a'
