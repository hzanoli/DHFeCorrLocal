def batch(sequence, n=1):
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
