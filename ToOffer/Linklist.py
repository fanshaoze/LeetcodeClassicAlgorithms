'''
题目：
输入一个链表的头节点，从尾到头反过来打印出每个节点的值。
'''


def PrintListReversingly(head):
    if head:
        if head.next:
            PrintListReversingly(head.next)
        print(head.val)


# 栈
def PrintListReversingly2(head):
    stack = []
    while head is not None:
        stack.append(head.val)
        head = head.next
    while stack:
        print(stack.pop())
    print()


'''
题目：链表中倒数第k个节点
输入一个链表，输出该链表中倒数第K的结点，为了符合大多数人的习惯，本题从1开始计数，即链表的尾结点是倒数第1个节点。
例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6，这个链表的倒数第3个节点是指为4的节点。
'''


def getN(head, N):
    a = 1
    h, n = head, head
    while h.next:
        h = h.next
        if a < N:
            a += 1
        else:
            n = n.next
    return n.val


'''
题目：如果一个链表中包含环，如何找出环的入口节点？例如，在如图所示的链表中，环的入口节点是节点3
         ___________
        |           |
1-->2-->3-->4-->5-->6
'''


def entry_node_of_loop(head):
    if head is None or head.next is None:
        return None
    slow_node = head.next
    fast_node = head.next.next
    while slow_node != fast_node:
        slow_node = slow_node.next
        fast_node = fast_node.next.next
    slow_node = head
    while slow_node != fast_node:
        slow_node = slow_node.next
        fast_node = fast_node.next
    return fast_node


'''
题目：定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
'''


def reverse(head):
    pre = None
    while head is not None:
        next_node = head.next
        head.next = pre
        pre = head
        head = next_node
    return pre


'''
题目：输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。例如，输入如下的链表1和链表2，则合并之后的升序链表如链表3所示。
链表1：1-->3-->5-->7
链表2：2-->4-->6-->8
链表3：1-->2-->3-->4-->5-->6-->7-->8
'''
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

'''
题目：输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。例如，输入如下的链表1和链表2，则合并之后的升序链表如链表3所示。
链表1：1-->3-->5-->7
链表2：2-->4-->6-->8
链表3：1-->2-->3-->4-->5-->6-->7-->8
'''


def mergeTwoLists(l1, l2):
    # write your code here
    if l1 is None:
        return l2
    if l2 is None:
        return l1
    if l1.val < l2.val:
        merge_head_node = l1
        merge_head_node.next = mergeTwoLists(l1.next, l2)
    else:
        merge_head_node = l2
        merge_head_node.next = mergeTwoLists(l1, l2.next)
    return merge_head_node


'''
题目：输入两颗二叉树A和B，判断B是不是A的子结构。
       1                3
      / \              /
A = 2   3      B =  4
        /
       4
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


def has_subtree(T1, T2):
    # write your code here
    result = False
    result = does_tree1_have_tree2(T1, T2)
    if not result:
        result = has_subtree(T1.left, T2)
    if not result:
        result = has_subtree(T1.right, T2)
    return result


def does_tree1_have_tree2(T1, T2):
    if T2 is None:
        return True
    elif T1 is None:
        return False
    elif T1.val != T2.val:
        return False
    else:
        a = does_tree1_have_tree2(T1.left, T2.left)
        b = does_tree1_have_tree2(T1.right, T2.right)
    return a and b


'''
题目：请完成一个函数，输入一颗二叉树，该函数输出它的镜像。
'''


def mirror_recursively(root):
    if root is None:
        return
    root.left, root.right = root.right, root.left
    if root.left is not None:
        mirror_recursively(root.left)
    if root.right is not None:
        mirror_recursively(root.right)


'''
题目：请实现一个函数，用来判断一颗二叉树是不是对称的，如果一颗二叉树和它的镜像一样，那么它是对称的。
'''


def is_sys(root):
    if not root:
        return False
    else:
        return sys(root.left, root.right)


def sys(root0, root1):
    if root0 is None and root1 is None:
        return True
    elif root0 is not None:
        return False
    elif root1 is not None:
        return False
    elif root0.val != root1.val:
        return False
    else:
        return sys(root0.left, root1.right) and sys(root1.left, root0.right)


'''
题目：输入两个链表，找出它们的第一个公共节点。
'''
def find_first_common_node(headA, headB):
    node1, node2 = headA, headB
    len1, len2 = 0, 0
    while node1 is not None:
        node1 = node1.next
        len1 += 1
    while node2 is not None:
        node2 = node2.next
        len2 += 1
    node1, node2 = headA, headB

    while len1 > len2:
        node1 = node1.next
        len1 -= 1
    while len2 > len1:
        node2 = node2.next
        len2 -= 1

    while node1 is not node2:
        node1 = node1.next
        node2 = node2.next
    return node1
