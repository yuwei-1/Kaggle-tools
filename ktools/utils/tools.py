def encode_in_order(array):
    d = {}
    idx = 0
    for i, n in enumerate(array):
        if n not in d:
            d[n] = idx
            idx += 1
        array[i] = d[array[i]]
    return array