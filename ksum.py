class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def kSum(nums, target, k):
            res = []
            # If we have run out of numbers to add, return res.
            if not nums:
                return res
            average_value = target // k
            if average_value < nums[0] or nums[-1] < average_value:
                return res
            if k == 2:
                return twoSum(nums, target)
            for i in range(len(nums)):
                # get the k-1 sum, add nums[i] as k sum
                if i == 0 or nums[i - 1] != nums[i]:
                    for subset in kSum(nums[i + 1:], target - nums[i], k - 1):
                        res.append([nums[i]] + subset)
            return res

        def twoSum(nums, target):
            res = []
            s = set()  # s to store the seen values (no diff)
            for i in range(len(nums)):
                if len(res) == 0 or res[-1][1] != nums[i]:
                    if target - nums[i] in s:
                        res.append([target - nums[i], nums[i]])
                s.add(nums[i])
            return res
        nums.sort()
        return kSum(nums, target, 4)
