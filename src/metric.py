import numpy as np
import pandas as pd


def accuracy_at_n(df, n=1):
    n_journals = df.journal_id.unique().shape[0]
    def calc_index_per_test_article(df_group):
        df_group = df_group.sort_values(by='score', ascending=False).reset_index(drop=True)
        index = df_group[df_group.journal_id == df_group.true_journal].index
        if index.shape[0] == 0:
            return n_journals
        index = index[0]+1
        return index
    basecase = df[df.modified_journal == '']
    series = basecase.groupby('doi').apply(calc_index_per_test_article)
    baseline_acc = (series<=n).mean()
    baseline_std = (series<=n).std(ddof=0)

    accs = []
    overall_per_test_article = []
    stds = []
    for journal in  df[df.modified_journal != ''].modified_journal.unique():
        series = df[df.modified_journal == journal].groupby('doi').apply(calc_index_per_test_article) <= n
        assert series.mean() == series.sum()/series.shape[0]
        accs.append(series.mean())
        stds.append(series.std(ddof=0))
        overall_per_test_article.extend(list(series))

    return {
        'mean baseline_acc': baseline_acc, 
        'mean mod_acc': np.mean(accs), 
        'std baseline_acc': baseline_std, 
        'std mod_acc': np.std(accs), 
        'overall mean mod_acc': np.mean(overall_per_test_article),
        'overall std mod_acc': np.std(overall_per_test_article),
    }



def shift_of_rank(df):
    basecase = df[df.modified_journal == '']
    n_journals = df.journal_id.unique().shape[0]
    def calc_index_per_test_article(df_group, mod_journal=None):
        df_group = df_group.sort_values(by='score', ascending=False).reset_index(drop=True)
        if df_group.modified_journal[0] == '':  # basecase
            index = df_group[df_group.journal_id == mod_journal].index
        else:
            index = df_group[df_group.journal_id == df_group.modified_journal].index
        if index.shape[0] == 0:
            return n_journals
        index = index[0]+1
        return index

    shifts = []
    shifts_all = []
    shifts_std = []
    for journal in df[df.modified_journal != ''].modified_journal.unique():
        series_mod = df[df.modified_journal == journal].groupby('doi').apply(calc_index_per_test_article)
        series_base = basecase.groupby('doi').apply(calc_index_per_test_article, journal)
        shift = (series_mod-series_base).mean()
        shifts_std.append((series_mod-series_base).std(ddof=0))
        shifts_all.extend(list(series_mod-series_base))
        shifts.append(shift)
    return {
        'mean shift': np.mean(shifts), 
        'std shift': np.std(shifts),
        'overall mean shift': np.mean(shifts_all),
        'overall std shift': np.std(shifts_all) 
    }


def test_shift_of_rank_simple():
    df = pd.DataFrame([
        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': ''},
        {'doi': 'x', 'method': 'x', 'score':'2', 'journal_id': 'b', 'modified_journal': ''},
        {'doi': 'x', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': ''},
        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'b'},
        {'doi': 'x', 'method': 'x', 'score':'4', 'journal_id': 'b', 'modified_journal': 'b'},
        {'doi': 'x', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': 'b'},
    ])
    result = shift_of_rank(df)
    assert result['mean shift'] == -1


def test_shift_of_rank_complex():
    df = pd.DataFrame([
        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': ''},
        {'doi': 'x', 'method': 'x', 'score':'2', 'journal_id': 'b', 'modified_journal': ''},
        {'doi': 'x', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': ''},

        {'doi': 'y', 'method': 'x', 'score':'1', 'journal_id': 'a', 'modified_journal': ''},
        {'doi': 'y', 'method': 'x', 'score':'3', 'journal_id': 'b', 'modified_journal': ''},
        {'doi': 'y', 'method': 'x', 'score':'9', 'journal_id': 'c', 'modified_journal': ''},

        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'b'},
        {'doi': 'x', 'method': 'x', 'score':'4', 'journal_id': 'b', 'modified_journal': 'b'}, # shift -1
        {'doi': 'x', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': 'b'},

        {'doi': 'y', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'b'},
        {'doi': 'y', 'method': 'x', 'score':'4', 'journal_id': 'b', 'modified_journal': 'b'},  # shift 0
        {'doi': 'y', 'method': 'x', 'score':'5', 'journal_id': 'c', 'modified_journal': 'b'},

        {'doi': 'y', 'method': 'x', 'score':'1', 'journal_id': 'a', 'modified_journal': 'c'},
        {'doi': 'y', 'method': 'x', 'score':'9', 'journal_id': 'b', 'modified_journal': 'c'},  # shift +1
        {'doi': 'y', 'method': 'x', 'score':'2', 'journal_id': 'c', 'modified_journal': 'c'},

        {'doi': 'x', 'method': 'x', 'score':'8', 'journal_id': 'a', 'modified_journal': 'c'},
        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'b', 'modified_journal': 'c'},  # shift -1
        {'doi': 'x', 'method': 'x', 'score':'7', 'journal_id': 'c', 'modified_journal': 'c'},
    ])
    result =  shift_of_rank(df)
    assert result['mean shift'] == -0.25
    assert result['std shift'] == 0.25
    assert result['overall mean shift'] == -0.25
    assert result['overall std shift'] == 0.82915619758885


def test_accuracy_at_n_1():
    df = pd.DataFrame([
        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': '', 'true_journal': 'a'}, # correct
        {'doi': 'x', 'method': 'x', 'score':'2', 'journal_id': 'b', 'modified_journal': '', 'true_journal': 'a'},
        {'doi': 'x', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': '', 'true_journal': 'a'},

        {'doi': 'y', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': '', 'true_journal': 'b'},
        {'doi': 'y', 'method': 'x', 'score':'5', 'journal_id': 'b', 'modified_journal': '', 'true_journal': 'b'}, # correct
        {'doi': 'y', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': '', 'true_journal': 'b'},

        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'b', 'true_journal': 'c'},
        {'doi': 'x', 'method': 'x', 'score':'4', 'journal_id': 'b', 'modified_journal': 'b', 'true_journal': 'c'}, # wrong
        {'doi': 'x', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': 'b', 'true_journal': 'c'},

        {'doi': 'y', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'b', 'true_journal': 'b'},
        {'doi': 'y', 'method': 'x', 'score':'4', 'journal_id': 'b', 'modified_journal': 'b', 'true_journal': 'b'}, # correct
        {'doi': 'y', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': 'b', 'true_journal': 'b'},

        {'doi': 'x', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'd', 'true_journal': 'c'},
        {'doi': 'x', 'method': 'x', 'score':'0', 'journal_id': 'b', 'modified_journal': 'd', 'true_journal': 'c'},
        {'doi': 'x', 'method': 'x', 'score':'5', 'journal_id': 'c', 'modified_journal': 'd', 'true_journal': 'c'}, # correct

        {'doi': 'y', 'method': 'x', 'score':'3', 'journal_id': 'a', 'modified_journal': 'd', 'true_journal': 'b'},
        {'doi': 'y', 'method': 'x', 'score':'4', 'journal_id': 'b', 'modified_journal': 'd', 'true_journal': 'b'}, # correct
        {'doi': 'y', 'method': 'x', 'score':'1', 'journal_id': 'c', 'modified_journal': 'd', 'true_journal': 'b'},
    ])
    result = accuracy_at_n(df)
    assert result['mean baseline_acc'] == 1
    assert result['mean mod_acc'] == 0.75
    assert result['std baseline_acc'] == 0.0
    assert result['std mod_acc'] == 0.25
    assert result['overall mean mod_acc'] ==  0.75
    assert result['overall std mod_acc'] == 0.4330127018922193
