# Sorting
import random

_list = [5, 2, 3, 6, 1, 4]


def bubble_sort(_list):
    """
    best: O(n) if sorted
    worst: O(n^2) if reverse sorted
    space: O(1)
    """
    n = len(_list)
    for i in range(n - 1, -1, -1):
        for j in range(i):
            if _list[j] > _list[j + 1]:
                _list[j], _list[j + 1] = _list[j + 1], _list[j]
    print(_list)


bubble_sort(_list)


def selection_sort(_list):
    """
    best: n, sorted
    worst: n^2, reverse sorted
    space: inspace
    """
    for i in range(len(_list) - 1):
        cur_id = i
        for j in range(i, len(_list)):
            if _list[j] < _list[cur_id]:
                cur_id = j
        _list[cur_id], _list[i] = _list[i], _list[cur_id]
    print(_list)


selection_sort(_list)


def insertion_sort(_list):
    """
    best: n sorted
    worst: n*2 reverse sorted
    """
    n = len(_list)
    for i in range(1, n):
        j = i
        val = _list[i]
        while j > 0 and val < _list[j - 1]:
            _list[j] = _list[j - 1]
            j -= 1
        _list[j] = val
    print(_list)


insertion_sort(_list)


def merge_sort(_list):
    """
    best:
    worst:
    space: O(n)
    """
    if len(_list) <= 1:
        return _list
    mid = len(_list) // 2
    l0 = merge_sort(_list[0:mid])
    l1 = merge_sort(_list[mid:len(_list)])
    return merge(l0, l1)


def merge(list0, list1):
    nums = []
    i, j = 0, 0
    n0, n1 = len(list0), len(list1)
    while i < n0 and j < n1:
        if list0[i] < list1[j]:
            nums.append(list0[i])
            i += 1
        else:
            nums.append(list1[j])
            j += 1
    while i < n0:
        nums.append(list0[i])
        i += 1
    while j < n1:
        nums.append(list1[j])
        j += 1
    return nums


print(merge_sort(_list))


def quick_sort(_list, l, r):
    """
    best: O(nlogn) always select middle
    worst: O(n^2) select the final position for each
    space: O(1)
    """
    if l < r:
        i = rand_partition(_list, l, r)
        quick_sort(_list, l, i)
        quick_sort(_list, i + 1, r)
    return _list


def rand_partition(_list, l, r):
    rand = random.randint(l, r)
    _list[l], _list[rand] = _list[rand], _list[l]
    i = partition(_list, l, r)
    return i


def partition(_list, l, r):
    val = _list[l]
    i, j = l, r
    while i < j:
        while i < j and _list[i] < val:
            i += 1
        while i < j and _list[j] > val:
            j -= 1
        _list[i], _list[j] = _list[j], _list[i]
    _list[i], _list[l] = _list[l], _list[i]
    return i


print(quick_sort(_list, 0, len(_list) - 1))


def radixSort(_list):
    """
    from back to front
    time: O(n*k)
    space: O(n+k)
    """
    size = len(str(max(_list)))
    for i in range(size):
        buckets = [[] for _ in range(10)]
        for num in _list:
            buckets[num // (10 ** i) % 10].append(num)
        _list.clear()
        for bucket in buckets:
            for num in bucket:
                _list.append(num)
        print(_list)
    return _list


_list = [692, 924, 969, 503, 871]
radixSort(_list)
