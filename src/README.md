Code for reproducing the results from the paper "Can editorial decision impair journal recommendations?  Analysing the impact of journal characteristics on recommendation systems".

Steps to run the code:
- Install the requirements via `pip install -r requirements.txt`
- Test via `pytest`
- Download the title abstract and journal from dimensions. Only the DOIs are given in the data directory.
- Run `main.py`


To reproduce the B!SON results:
- Install the [B!SON backend](https://gitlab.com/TIBHannover/bison/yak) requirements and load the data as described in their README
- Generate number of articles per journal file or use the one in the repo. It is create via PostgreSQL command-line tool and the query:
```
\COPY (SELECT rj.idx, count(*) FROM recommender_journal rj 
       JOIN recommender_article ra ON rj.pissn = ANY(ra.journal_issns) OR rj.eissn = ANY(ra.journal_issns) 
       GROUP BY rj.idx
       ORDER BY COUNT(*))
TO 'articles_per_journal.csv' CSV;
```
- Run `bison_evaluate.py`
- Change sum to mean in `recommender/search.py`
- Run `python manage.py generate_score_model`
- Restart ray serve
- Run `bison_evaluate.py` again
