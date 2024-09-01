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


# stack but linklist
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class Stack:
    def __init__(self):
        self.top = None

    def is_empty(self):
        return self.top is None

    def insert(self, v):
        node = Node(v)
        node.next = self.top
        self.top = node

    def pop(self):
        if self.is_empty():
            return None
        else:
            cur = self.top
            self.top = self.top.next
            return cur

# monotone Stack
def monotoneIncreasingStack(nums):
    stack = []
    for num in nums:
        while stack and num >= stack[-1]:
            stack.pop()
        stack.append(num)


# stack: list pop from tail, list append from tail

# queue: list append from tail, pop with list.pop(0)
if __name__ == '__main__':
    main()
