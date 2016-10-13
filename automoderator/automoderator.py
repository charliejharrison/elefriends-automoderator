"""
Main code for automoderator.  Contains NLTK scikit-learn pipeline.
"""

__author__ = 'wah'


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB

# from sklearn.cross_validation import cross_val_score

from transformers import *
    #ColumnDifference, Cosine, DatetimeToValue, DatetimeToTimestamp
from preprocess import load_data


from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import metrics


def main(verbose=False):
    """
    Do column mappings:
    #  - datetime -> timestamp (function transformer); normalise these?
    #  - vectorize text
    #  - [date_joined, datetime] -> two-column diff of timestamps
    #  - datetime -> cosine(2*pi*hour / 24)
    """
    if verbose:
        print("Loading data...")
    X, y = load_data(path='../data/posts.features.01.csv', labels='flag_2',
                     content_types='Post', boolean_labels=True)

    # Lists of transformers passed to DataFrameMapper are applied sequentially
    df_mapper = DataFrameMapper([
        (['datetime', 'date_joined'], DatetimeToTimestamp()),
        (
            ['date_joined', 'datetime'],
            [DatetimeToTimestamp('timestamp'), ColumnDifference()]
        ),
        ('datetime', [DatetimeToValue('hours'), Cosine()]),
        ('content_body', CountVectorizer()),
        (['contains_video', 'contains_image', 'contains_file',
          'contains_link', 'removed_user', 'removed_moderator'], None)
    ])

    pipeline = Pipeline([('dfm', df_mapper), ('clf', MultinomialNB())])

    #TODO: filter out content types?
    #TODO: tune the vectorizer:
    #   - try Count and TfIdf
    #   - try different range of ngrams
    #   - try thresholds for word frequency

    if verbose:
        print("Calculating cross validation scores...")
    # scores = cross_val_score(pipeline, X, y.iloc[:, 0], n_jobs=-1, verbose=True)
    scores = cross_val_score(pipeline, X, y.iloc[:, 0], verbose=True)

    if verbose:
        print("Done! Scores:")
        print(scores)


if __name__ == '__main__':
    main(verbose=True)