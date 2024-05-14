import pandas as pd
import pytest
from textblob import TextBlob

RANDOM_STATE=42


def many_articles(df, n_low, n_high):
    print('many articles')
    df = df.copy()
    df['modified_journal_id'] = ''

    #base case
    cases = [pd.concat([
        df[df.type=='train'].groupby('journal_id').sample(n_low, random_state=RANDOM_STATE),
        df[df.type=='test']
    ])]

    #modified cases
    for journal in df.journal_id.unique():
        df_mod = df.copy()
        df_mod.modified_journal_id = journal
        df_mod_jour = df_mod[(df.type=='train') & (df.journal_id==journal)].sample(n_high, random_state=RANDOM_STATE)
        df_mod_rest = df_mod[(df.type=='train') & (df.journal_id!=journal)].groupby('journal_id').apply(lambda g: g.sample(n_low, random_state=RANDOM_STATE))
        df_mod_test = df_mod[df.type=='test']
        cases.append(pd.concat([df_mod_test, df_mod_jour, df_mod_rest], ignore_index=True))
    return cases


def long_title(df, n_articles, len_text_low, len_text_high):
    print('long title')
    df = df.copy()
    df['modified_journal_id'] = ''

    #base case
    cases = [pd.concat([
        df[(df.type=='train') & (df.title.str.len()<=len_text_low)].groupby('journal_id').sample(n_articles, random_state=RANDOM_STATE),
        df[df.type=='test']
    ])]

    #modified cases
    for journal in df.journal_id.unique():
        df_mod = df.copy()
        df_mod.modified_journal_id = journal

        df_mod_jour = df_mod[
            (df.type=='train') & 
            (df.journal_id==journal) & 
            (df.title.str.len()>=len_text_high)
        ].sample(n_articles, random_state=RANDOM_STATE)

        df_mod_rest = df_mod[
            (df.type=='train') & 
            (df.journal_id!=journal) & 
            (df.title.str.len()<=len_text_low)
        ].groupby('journal_id').apply(lambda g: g.sample(n_articles, random_state=RANDOM_STATE))

        df_mod_test = df_mod[df.type=='test']
        df_mod = pd.concat([df_mod_test, df_mod_jour, df_mod_rest], ignore_index=True)
        assert df_mod[df_mod.type == 'train'].journal_id.nunique() == df.journal_id.nunique()
        assert (df_mod[df_mod.type == 'train'].groupby('journal_id').size() == n_articles).all() == True
        cases.append(df_mod)
    return cases


def long_abstract(df, n_articles, len_text_low, len_text_high):
    print('long abstract')
    df = df.copy()
    df['modified_journal_id'] = ''
    df = df.dropna(subset=['abstract'])
    abstract_length = df.abstract.apply(lambda r: len(TextBlob(r).words))

    #base case
    cases = [pd.concat([
        df[(df.type=='train') & (abstract_length <= len_text_low)].groupby('journal_id').sample(n_articles, random_state=RANDOM_STATE),
        df[df.type=='test']
    ])]

    #modified cases
    for journal in df.journal_id.unique():
        df_mod = df.copy()
        df_mod.modified_journal_id = journal

        df_mod_jour = df_mod[
            (df.type=='train') & 
            (df.journal_id==journal) & 
            (abstract_length>=len_text_high)
        ]
        df_mod_jour = df_mod_jour.sample(n_articles, random_state=RANDOM_STATE)

        df_mod_rest = df_mod[
            (df.type=='train') & 
            (df.journal_id!=journal) & 
            (abstract_length<=len_text_low)
        ].groupby('journal_id').apply(lambda g: g.sample(n_articles, random_state=RANDOM_STATE))

        df_mod_test = df_mod[df.type=='test']

        df_mod = pd.concat([df_mod_test, df_mod_jour, df_mod_rest], ignore_index=True)
        assert df_mod[df_mod.type == 'train'].journal_id.nunique() == df.journal_id.nunique()
        assert (df_mod[df_mod.type == 'train'].groupby('journal_id').size() == n_articles).all() == True
        cases.append(df_mod)
    return cases


def test_long_abstract():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'abstract': 'Very short'},
        {'type': 'train', 'journal_id': 'x', 'abstract': 'This is much much longer text'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Muy corto'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Too much text for anyone to read'},
        {'type': 'test', 'journal_id': 'y', 'abstract': 'This is very long and long and long'},
        {'type': 'test', 'journal_id': 'x', 'abstract': 'This'},
    ])
    cases = long_abstract(df, 1, 2, 5)
    assert len(cases) == 3
    assert cases[0].shape[0] == 4
    assert cases[0].modified_journal_id[0] == ''
    assert (cases[0][cases[0].type == 'train'].abstract.str.len() < 15).all() == True
    assert cases[-1][cases[-1].type == 'test'].shape[0] == 2
    assert cases[-1][cases[-1].type == 'train'].shape[0] == 2
    assert (cases[-1][(cases[-1].type == 'train') & (cases[-1].journal_id == cases[-1].modified_journal_id[0])].abstract.str.len() > 20).all() == True
    assert (cases[-1][(cases[-1].type == 'train') & (cases[-1].journal_id != cases[-1].modified_journal_id[0])].abstract.str.len() < 20).all() == True


def test_long_abstract_not_enough():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'abstract': 'This is much much longer text'},
        {'type': 'train', 'journal_id': 'y', 'abstract': 'Muy corto'},
        {'type': 'test', 'journal_id': 'y', 'abstract': 'This is very long and long and long'},
        {'type': 'test', 'journal_id': 'x', 'abstract': 'This'},
    ])
    with pytest.raises(Exception):
        long_abstract(df, 1, 1, 5)
        

def test_long_title():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'title': 'Kurz'},
        {'type': 'train', 'journal_id': 'x', 'title': 'Very long'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Short'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Seeehr lang'},
        {'type': 'test', 'journal_id': 'y', 'title': 'This is long'},
        {'type': 'test', 'journal_id': 'x', 'title': 'This'},
    ])
    cases = long_title(df, 1, 5, 6)
    assert len(cases) == 3
    assert cases[0].shape[0] == 4
    assert cases[0].modified_journal_id[0] == ''
    assert cases[-1].modified_journal_id[0] in ['x', 'y']
    assert (cases[0][cases[0].type == 'train'].title.str.len() <= 5).all() == True
    assert cases[-1][cases[-1].type == 'test'].shape[0] == 2
    assert cases[-1][cases[-1].type == 'train'].shape[0] == 2
    assert (cases[-1][(cases[-1].type == 'train') & (cases[-1].journal_id == cases[-1].modified_journal_id[0])].title.str.len() >= 6).all() == True
    assert (cases[-1][(cases[-1].type == 'train') & (cases[-1].journal_id != cases[-1].modified_journal_id[0])].title.str.len() <= 5).all() == True


def test_long_title_not_enough():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x', 'title': 'This is much much longer text'},
        {'type': 'train', 'journal_id': 'y', 'title': 'Muy corto'},
        {'type': 'test', 'journal_id': 'y', 'title': 'This is very long and long and long'},
        {'type': 'test', 'journal_id': 'x', 'title': 'This'},
    ])
    with pytest.raises(Exception):
        x = long_title(df, 1, 5, 5)


def test_many_articles():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x'},
        {'type': 'train', 'journal_id': 'x'},
        {'type': 'train', 'journal_id': 'x'},
        {'type': 'train', 'journal_id': 'y'},
        {'type': 'train', 'journal_id': 'y'},
        {'type': 'test', 'journal_id': 'x'},
        {'type': 'test', 'journal_id': 'y'},
    ])
    cases = many_articles(df, 1, 2)
    assert len(cases) == 3
    assert cases[0].shape[0] == 4
    assert cases[0].modified_journal_id[0] == ''
    assert cases[-1][cases[-1].type == 'test'].shape[0] == 2
    assert cases[-1][cases[-1].type == 'train'].shape[0] == 3
    assert cases[-1][(cases[-1].type == 'train') & (cases[-1].journal_id == cases[-1].modified_journal_id[0])].shape[0] == 2
    assert cases[-1][(cases[-1].type == 'train') & (cases[-1].journal_id != cases[-1].modified_journal_id[0])].shape[0] == 1


def test_many_articles_not_enough():
    df = pd.DataFrame([
        {'type': 'train', 'journal_id': 'x'},
        {'type': 'train', 'journal_id': 'x'},
        {'type': 'train', 'journal_id': 'y'},
        {'type': 'test', 'journal_id': 'x'},
        {'type': 'test', 'journal_id': 'y'},
    ])
    with pytest.raises(Exception):
        many_articles(df, 1, 2)
