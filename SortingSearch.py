def partition(nums, l, r):
    i = l
    while l < r:
        if nums[l] < nums[r]:  # i maintain the most right larger one
            nums[l], nums[i] = nums[i], nums[l]
            i += 1
        l += 1
    nums[i], nums[r] = nums[r], nums[i]
    print(nums)


def binary_search(sorted_nums, l, r, val):
    while l <= r:  # must consider equal
        mid = (l + r) // 2
        if sorted_nums[mid] == val:
            return mid
        elif sorted_nums[mid] > val:
            r = mid - 1
            binary_search(sorted_nums, l, r, val)
        else:
            l = mid + 1
            binary_search(sorted_nums, l, r, val)
    return None


partition([0, 10, 8, 7, 3, 9, 6, 5], 0, 7)
print(binary_search([i for i in range(10)], 0, 7, 5))
