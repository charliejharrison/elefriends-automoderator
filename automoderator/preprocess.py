import datetime
from pandas import read_csv
from numpy import iterable as isiterable


def get_by_val_or_ind(ar, wanted=None):
    """
    Select elements from an array by value or by index, singly or in an
    iterable.

    Gotchas:
      - does not handle mixtures of values and indices
      - integer values will be treated as indices
      - if a single value is passed, and this is not found in the array,
      it will fail.  If multiple values are passed and any of them are found,
      they will be returned - values not found are ignored silently

    :param ar:
    :param wanted:
    :return: list of elements that is either the intersection of wanted and
    ar, or the elements of ar at the indices specified in wanted.  Single
    elements will also be returned in a list
    """
    # Better logic?:
    # cts = {'Message', 'Comment', 'Post'}
    # if content_types:
    #     if isiterable(content_types) and not isinstance(content_types, str):
    #         content_types = set(content_types)
    #     else:
    #         content_types = {content_types}
    #     content_types = content_types.intersection(cts)
    #     if not content_types:
    #         raise ValueError("Unrecognised content types")
    # data = data[data['content_type'].isin(content_types)]

    if isiterable(wanted) and not isinstance(wanted, str):
        if isinstance(wanted[0], int):
            return [ar[i] for i in wanted]
        else:
            result = [x for x in wanted if x in set(ar)]
            if result:
                return result
            else:
                raise ValueError("Could not find any of the passed values")
    elif isinstance(wanted, int):
        return [ar[wanted]]
    elif wanted in ar:
        return [wanted]
    else:
        raise ValueError("Unrecognised argument")


def load_data(path='../data/features.01.csv', labels=None,
              boolean_labels=True,
              content_types=None,
              unflagged_only=True):
    """
    Load the feature data from file.

    :param path:
    :param labels:
    :return:
    """
    ##TODO: set this up as a Transformer to allow experimenting with
    # alternative null classes, multilabel, etc.

    data_types = {'removed': bool,
                  'removed_user': bool,
                  'removed_moderator': bool}

    data = read_csv(path,
                    dtype=data_types,
                    true_values=['t'],
                    false_values=['f'],
                    parse_dates=['datetime', 'date_joined'])

    # Filter out content types
    cts = ['Message', 'Comment', 'Post']
    if content_types:
        content_types = get_by_val_or_ind(cts, content_types)
        data = data[data['content_type'].isin(content_types)]

    # Extract labels
    label_cols = ['flag_1', 'flag_2', 'flag_3', 'flag_4', 'flag_5', 'flag_6',
                  'flag_7', 'flag_8']
    if labels:
        labels = get_by_val_or_ind(label_cols, labels)
    else:
        labels = label_cols

    # TODO: more sophisticated handling of unflagged_only option
    # Only compare against content with no flags
    if unflagged_only:
        data['any_flags'] = data[label_cols].any(axis=1)
        data = data[~data['any_flags'] | data[labels].any(axis=1)]

    feature_cols = [c for c in data.columns if c not in set(label_cols)]

    labels = data[labels]
    if boolean_labels:
        labels = labels > 0

    return data[feature_cols], labels
