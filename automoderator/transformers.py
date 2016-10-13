"""
Module containing custom transformers
"""

from sklearn.base import TransformerMixin
from pandas import DataFrame
import numpy as np
from numpy import datetime64 as dt64, timedelta64 as td64
from math import cos, pi as PI


class DatetimeToTimestamp(TransformerMixin):
    """
    Convert datetime to seconds since epoch (seconds since 1970-01-01T00:00:00)
    """
    def __init__(self, *args, **kwargs):
        self._dt_func = lambda x: x.timestamp()
        self._nd_func = lambda x: (x.astype('datetime64[s]') - dt64(0, 's')) / \
                            td64(1, 's')

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, value='days', *args, **kwargs):
        if isinstance(X, DataFrame):
            return X.applymap(self._dt_func)
        elif isinstance(X, np.ndarray):
            return self._nd_func(X)
        else:
            raise TypeError("DatetimeToTimestamp requires DataFrame or ndarray "
                            "(not {})".format(type(X)))


class DatetimeToValue(TransformerMixin):
    """
    Extract a second, minute, hour, day, month, or year value from a dataframe
    containing datetime objects.)
    """
    def __init__(self, value='days'):
        """
        Compile the lambda functions for extracting the appropriate value
        from a DataFrame or ndarray
        """
        if value in {'seconds', 'second', 's'}:
            self._df_func = lambda x: x.second
            self._np_func = lambda x: (x.astype('datetime64[s]') -
                             x.astype('datetime64[m]')) / td64(1, 's')
        elif value in {'minutes', 'minute', 'm'}:
            self._df_func = lambda x: x.minute
            self._np_func = lambda x: (x.astype('datetime64[m]') -
                             x.astype('datetime64[h]')) / td64(1, 'm')
        elif value in {'hours', 'hour', 'h'}:
            self._df_func = lambda x: x.hour
            self._np_func = lambda x: (x.astype('datetime64[h]') -
                             x.astype('datetime64[D]')) / td64(1, 'h')
        elif value in {'days', 'day', 'd'}:
            self._df_func = lambda x: x.day
            self._np_func = lambda x: (x.astype('datetime64[D]') -
                             x.astype('datetime64[M]') + td64(1, 'D')) / \
                td64(1, 'D')
        elif value in {'months', 'month', 'M'}:
            self._df_func = lambda x: x.month
            self._np_func = lambda x: (x.astype('datetime64[M]') -
                             x.astype('datetime64[Y]') + td64(1, 'M')) / \
                td64(1, 'M')
        elif value in {'years', 'year', 'y'}:
            self._df_func = lambda x: x.year
            self._np_func = lambda x: (x.astype('datetime64[Y]') -
                             dt64('0', 'Y')) / td64(1, 'Y')
        else:
            raise ValueError("value must be one of ['seconds', 'minutes', "
                             "'hours', 'days', 'years'] (not '{}')"
                             .format(value))
        # self._np_func = np.vectorize(npf, otypes=[np.int])

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, value='days', *args, **kwargs):
        if isinstance(X, DataFrame):
            return X.applymap(self._df_func)
        elif isinstance(X, np.ndarray):
            return self._np_func(X)
        else:
            raise TypeError("DatetimeToValue requires DataFrame or ndarray (not"
                            "{})".format(type(X)))


class ColumnDifference(TransformerMixin):
    """
    Transformer for finding the difference (object.__sub__) between two or more
    columns.

    The first column will be used as the baseline; transform() returns the
    result of (col - baseline) for each additional column.
    """
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        if isinstance(X, DataFrame):
            return X.iloc[:, 1:].sub(X.iloc[:, 0], axis=0)
        elif isinstance(X, np.ndarray):
            return X[:, 1:] - X[:, 0:1]
        else:
            raise TypeError("ColumnDifference expects a DataFrame or ndarray "
                            "(not {})".format(type(X)))


class Cosine(TransformerMixin):
    """
    Take the cosine of an input value, returning periodic value in [-1:1]
    Returns cos(val * 2*pi / period)
    """
    def __init__(self, period=2*PI):
        self.period = period

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        # if isinstance(X, DataFrame):
        #     return X.applymap(lambda x: cos(x * 2 * PI / self.period))
        # elif isinstance(X, np.ndarray):
        #     return np.cos(X * 2 * PI / self.period)
        # else:
        #     raise TypeError("Cosine expects a DataFrame or ndarray (not {})"
        #                     .format(type(X)))
        return np.cos(X * 2 * PI / self.period)