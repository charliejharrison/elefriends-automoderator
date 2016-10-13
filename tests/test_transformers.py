import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from automoderator import transformers
from pandas import DataFrame
from math import ceil, pi as PI
from datetime import datetime
from numpy import isclose, array_equal, ndarray


def test_dataframe_and_ndarray(trans, inp, exp, name=None, allow_close=False):
    """
    Take a transformer and two DataFrames.  Test the transformer works with
    a DataFrame and the equivalent nump.ndarray.

    For the sake of keeping tests DRY.

    :param trans: transformer instance
    :param inp: input (DataFrame)
    :param exp: expected output (DataFrame)
    :param name: Name of the class being tested
    :return:
    """
    if not name:
        name = type(trans).__name__

    out = None
    try:
        # Use numpy.isclose() to allow for floating point errors
        out = trans.fit_transform(inp)
        if allow_close:
            assert isclose(exp, out).all()
        else:
            assert out.equals(exp)

        exp = exp.as_matrix()
        out = trans.fit_transform(inp.as_matrix())
        if allow_close:
            assert isclose(out, exp).all()
        else:
            assert array_equal(out, exp)
    except AssertionError as err:
        print("{} failed:\n\tExpected:\n\n{}\n\n\tGot:\n\n{}".format(name, exp,
                                                                 out))
        raise err


def test_ColumnDifference():
    trans = transformers.ColumnDifference()
    inp = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    exp = DataFrame({'b': [3, 3, 3]})
    test_dataframe_and_ndarray(trans, inp, exp, "ColumnDifference")


def test_Cosine():
    trans = transformers.Cosine(period=24)
    inp = DataFrame({'times': [0, 4, 6, 8, 12, 16, 18, 20, 24]})
    exp = DataFrame({'times': [1, 0.5, 0, -0.5, -1, -0.5, 0.0, 0.5, 1]})
    test_dataframe_and_ndarray(trans, inp, exp, "Cosine", allow_close=True)


def test_DatetimeToValue():
    inp = DataFrame({'dts': [datetime(1970, 1, 1, 1), datetime(1986, 1, 15, 12),
                             datetime(2016, 7, 15, 23)]})
    name = "DatetimeToValue"

    value = 'seconds'
    trans = transformers.DatetimeToValue(value)
    exp = DataFrame({'dts': [0, 0, 0]})
    test_dataframe_and_ndarray(trans, inp, exp, name + '_' + value)

    value = 'minute'
    trans = transformers.DatetimeToValue(value)
    exp = DataFrame({'dts': [0, 0, 0]})
    test_dataframe_and_ndarray(trans, inp, exp, name + value)

    # test default value='days'
    trans = transformers.DatetimeToValue()
    exp = DataFrame({'dts': [1, 15, 15]})
    test_dataframe_and_ndarray(trans, inp, exp, name + '_days')

    value = 'h'
    trans = transformers.DatetimeToValue(value)
    exp = DataFrame({'dts': [1, 12, 23]})
    test_dataframe_and_ndarray(trans, inp, exp, name + '_' + value)

    value = 'years'
    trans = transformers.DatetimeToValue(value)
    exp = DataFrame({'dts': [1970, 1986, 2016]})
    test_dataframe_and_ndarray(trans, inp, exp, name + '_' + value)


def test_DatetimeToTimestamp():
    inp = DataFrame({'dts': [datetime(1970, 1, 1, 1), datetime(1986, 1, 15, 12),
                             datetime(2016, 7, 15, 23)]})
    exp = DataFrame({'dts': [3.60000000e+03, 5.06174400e+08, 1.46862360e+09]})
    trans = transformers.DatetimeToTimestamp()
    test_dataframe_and_ndarray(trans, inp, exp, name='DatetimeToTimestamp')


if __name__ == '__main__':
    try:
        test_ColumnDifference()
        test_Cosine()
        test_DatetimeToValue()
        test_DatetimeToTimestamp()

        print("All test passed :)")

    except AssertionError as e:
        print("Tests failed :(")