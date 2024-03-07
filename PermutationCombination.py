from collections import Counter


def permuteUnique(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    permutations = []
    counter = Counter(nums)

    def findAllPermutations(res):
        if len(res) == len(nums):  # if meet
            permutations.append(res)
            return

        for key in counter:
            if counter[key]:
                counter[key] -= 1  # decrement visited key
                findAllPermutations(res + [key])
                counter[key] += 1  # restore the state of visited key to find the next path

    findAllPermutations([])
    return permutations


def next_permutation(l):
    n = len(l)
    if n <= 1:
        return False
    i = n - 1
    while True:
        j = i
        i -= 1
        if l[i] < l[j]:
            k = n - 1
            while not (l[i] < l[k]):
                k -= 1
            l[i], l[k] = l[k], l[i]
            l[j:] = reversed(l[j:])
            return True
        if i == 0:
            l.reverse()
            return False


print(permuteUnique([1, 1, 2]))
