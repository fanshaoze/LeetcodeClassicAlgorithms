

def nextLarger(nums):
    stack = []
    ret = [0] * len(nums)
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            ret[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    return ret


print(nextLarger([1, 4, 2, 3, 5]))
