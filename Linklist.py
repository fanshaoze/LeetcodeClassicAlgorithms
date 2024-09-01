# define linklist
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class linklist:
    def __init__(self):
        self.head = None


def create(vals):
    link = linklist()
    if not vals: return link
    link.head = ListNode(vals[0], None)
    tail = link.head
    for i in range(1, len(vals)):
        node = ListNode(vals[i], None)
        tail.next = node
        tail = tail.next
    return link


def length(link):
    len = 0
    node = link.head
    while node:
        len += 1
        node = node.next
    return len


def find(val, link):
    cur = link.head
    while cur:
        if val == cur.val:
            return cur
        cur = cur.next

    return None


def insertFront(val, link):
    node = ListNode(val)
    node.next = link.head
    link.head = node


def insertRear(val, link):
    cur = link.head
    while cur.next:
        cur = cur.next
    cur.next = ListNode(val, None)
    return link.head


def insertInside(link, index, val):
    count = 0
    cur = link.head
    while cur and count < index - 1:
        count += 1
        cur = cur.next

    if not cur:
        return 'Error'

    node = ListNode(val)
    node.next = cur.next
    cur.next = node


def delete(link, val):
    head = link.head
    while head and head.val == val:
        head = head.next
        link.head = head
    while head and head.next:
        if head.next.val == val:
            head.next = head.next.next
        else:
            head = head.next
    return link.head
