def batch(sequence, n=1):
    """Divide sequence into a list of lists, where each element of the new list contains n elementes.

    Parameters
    ----------
    sequence: iterable
        The values that will be combined

    n: int
        The number of elements that should be combined

    """
    if n == 0:
        yield sequence

    length = len(sequence)
    for ndx in range(0, length, n):
        yield sequence[ndx:min(ndx + n, length)]


def format_list_to_bash(string_list):
    merged = ''
    for x in string_list:
        merged += str(x) + ','
    return merged[:-1]
