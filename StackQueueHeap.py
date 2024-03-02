import heapq


def build_heap(vals):
    heap = []
    for val in vals:
        heapq.heappush(heap, val)
    return heap


def pop_heap(heap):
    return heapq.heappop(heap)


def adjust(heap):
    heapq.heapify(heap)
    return heap


def main():
    heap = build_heap([0, 1, 2, 3, 4, 5])
    print(heap)
    print(pop_heap(heap))
    print(heap)
    heap = [1, 5, 3, 7, 4, 7]
    print(adjust(heap))
    return


# stack: list pop from tail, list append from tail

# queue: list append from tail, pop with list.pop(0)
if __name__ == '__main__':
    main()
