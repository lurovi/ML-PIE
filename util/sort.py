def heapify(arr, ind, comparator, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and comparator(arr[i], arr[l]):
        largest = l

    if r < n and comparator(arr[largest], arr[r]):
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        ind[i], ind[largest] = ind[largest], ind[i]
        heapify(arr, ind, comparator, n, largest)


def heapsort(v, comparator, inplace=False, reverse=False):
    n = len(v)
    ind = list(range(n))
    if not(inplace):
        arr = [x for x in v]
    else:
        arr = v

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, ind, comparator, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        ind[i], ind[0] = ind[0], ind[i]
        heapify(arr, ind, comparator, i, 0)

    if not(reverse):
        return arr, ind
    else:
        return arr[::-1], ind[::-1]
