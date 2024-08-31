

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


def characterReplacement(s, k):
    """
    :type s: str
    :type k: int
    :rtype: int
    """
    # maintain the count of letter in current slide
    c_dic = {}
    for c in s: c_dic[c] = 0
    max_count = 0
    max_l = 0
    if len(s) == 0:
        return 0
    else:
        l = 0
        for r in range(len(s)):
            c_dic[s[r]] += 1
            max_count = max([max_count, c_dic[s[r]]])
            if max_count+k>=r-l+1:
                max_l = max([max_l, r-l+1])
            else:
                c_dic[s[l]] -= 1
                l+=1
    return max_l
