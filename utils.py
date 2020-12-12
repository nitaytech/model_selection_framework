import typing
from sklearn.model_selection import KFold


def iterable_not_str(item: typing.Any) -> bool:
    """
    :param item: Any
    :return: True if item is Iterable which is not str, otherwise False
    """
    if isinstance(item, typing.Iterable) and not isinstance(item, str):
        return True
    else:
        return False


def to_iterable(item: typing.Any, to_list: bool = False) -> typing.Iterable:
    """
    If item is not iterable, return a tuple containing the item
    :param item: Any
    :return: iterable item
    """
    if iterable_not_str(item):
        if to_list:
            return list(item)
        return item
    else:
        if to_list:
            return [item]
        return (item,)


def sequence_not_str(item: typing.Any) -> bool:
    """
    :param item: Any
    :return: True if item is Sequence which is not str, otherwise False
    """
    if isinstance(item, typing.Sequence) and not isinstance(item, str):
        return True
    else:
        return False


def kwargs_must_contain(must_contain_list: typing.Iterable[str], **kwargs) -> bool:
    must_contain_list = to_iterable(must_contain_list)
    for item in must_contain_list:
        if item not in kwargs:
            raise ValueError(f"{item} is missing from kwargs")
    return True


def deep_round(item: typing.Any, round_precision: int = 3):
    if isinstance(item, float):
        return round(item, round_precision)
    elif isinstance(item, typing.Dict):
        return {k: deep_round(v, round_precision) for k, v in item.items()}
    elif iterable_not_str(item):
        return [deep_round(v, round_precision) for v in item]
    elif isinstance(item, str):
        try:
            item = eval(item)
            return deep_round(item, round_precision)
        except Exception as e:
            return item
    else:
        return item


def dict_to_ordered_str(d: dict):
    s = '{'
    for k, v in sorted(d.items()):
        if isinstance(k, str):
            k = f"'{k}'"
        if isinstance(v, str):
            k = f"'{v}'"
        s += f"{k}: {v}, "
    s = s[:-2] + '}'
    return s


def k_folds_indices(n: int, k: int, random_state: int = 42, shuffle: bool = True):
    kf = KFold(n_splits=k, random_state=random_state, shuffle=shuffle)
    return [(train.tolist(), test.tolist()) for train, test in kf.split(list(range(n)))]