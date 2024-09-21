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
            r = mid + 1
        else:
            l = mid + 1
    return None


def binary_search_interval_left(nums, val):
    """
    find left bound
    :param nums:
    :param val:
    :return:
    """
    # if allowed outside the range of nums, deal with it before the while loop
    l = 0
    r = len(nums) - 1
    while True:
        mid = (l + r) // 2
        if val < nums[l]:
            r = mid - 1
        elif val > nums[l + 1]:
            l = mid + 1
        else:
            l = mid
            break
    return [l, l + 1]


# close the other side when target is equal to this side
def binary_search_interval_right(nums, val):
    """
    find right bound, denote with l
    :param nums:
    :param val:
    :return:
    """
    # if allowed outside the range of nums, deal with it before the while loop
    l = 0
    r = len(nums)
    while l < r:
        mid = (l + r) // 2
        if val > nums[l]:
            l = mid + 1
        else:
            r = mid
            break
    return [l - 1, l]


def binary_search_find_peak(nums):
    l = 0
    r = len(nums) - 1
    while True:
        mid = l + (r - l) // 2
        print(nums[mid])
        if mid < len(nums) - 1 and nums[mid + 1] > nums[mid]:
            l = mid + 1
        elif mid > 0 and nums[mid - 1] > nums[mid]:
            r = mid - 1
        else:
            return mid


def search_in_rotated(nums, target):
    l = 0
    r = len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] <= nums[r]:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
        elif nums[mid] >= nums[l]:
            if nums[mid] > target >= nums[l]:
                r = mid - 1
            else:
                l = mid + 1


partition([0, 10, 8, 7, 3, 9, 6, 5], 0, 7)
print(binary_search([i for i in range(10)], 0, 7, 5))
print(search_in_rotated([7, 8, 9, 0, 1, 2, 3, 4, 5, 6], 7))
#
# (0, 6, 3, 7, [3, 4, 6, 7])
# (0, 2, 1, 5, [2, 2, 3, 4])
#
# (0, 6, 3, 7, [3, 4, 6, 7])
# (0, 3, 1, 5, [2, 2, 3, 4])
# (2, 3, 2, 6, [2, 3, 4, 6])
