def dfs():
    pass


def bfs_level(root):
    queue = [root]
    while queue:
        tmp_q = []
        level = []
        while queue:
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                tmp_q.append(node.left)
            if node.right:
                tmp_q.append(node.right)
        queue = tmp_q


def bfs_level_2(root):
    # for + pop version
    queue = [root]
    while queue:
        for _ in range(len(queue)):
            current = queue.pop(0)
            if current and current.left:
                queue.append(current.left)
                queue.append(current.right)


def bfs_depth(target, k, ret, g):
    """
    bfs but record depth in queue
    :param target: target node
    :param k: distance
    :param ret: return
    :param g: graph
    :return:
    """
    visited = set()
    q = []
    q.append((target, 0))
    while q:
        n, lev = q.pop(0)
        visited.add(n)
        if lev == k:
            ret.append(n.val)
        for nei in g[n]:
            if nei not in visited:
                q.append((nei, lev + 1))  # record the level


def graph_row_column_dfs(grid, i, j, m, n):  # filled with '0' or '1'
    if i < 0 or j < 0 or i >= m or j >= n:
        return
    else:
        if grid[i][j] == '1':
            # print(i,j)
            grid[i][j] = '0'
            graph_row_column_dfs(grid, i - 1, j, m, n)  # pre may not be considered as search from dfs but not iteration
            graph_row_column_dfs(grid, i + 1, j, m, n)
            graph_row_column_dfs(grid, i, j - 1, m, n)
            graph_row_column_dfs(grid, i, j + 1, m, n)
        else:
            return


def dfs_search_path(i, j, board, m, n, word, idx):
    if i < 0 or i >= m or j < 0 or j >= n:
        return False
    else:
        if board[i][j] != word[idx]:
            return False
        else:
            tmp = board[i][j]
            board[i][j] = '.'
            if idx == len(word) - 1:
                return True
            else:
                t = dfs_search_path(i, j - 1, board, m, n, word, idx + 1)
                d = dfs_search_path(i, j + 1, board, m, n, word, idx + 1)
                l = dfs_search_path(i - 1, j, board, m, n, word, idx + 1)
                r = dfs_search_path(i + 1, j, board, m, n, word, idx + 1)
                board[i][j] = tmp
                if t or d or l or r:
                    return True


def dfs_count_sum(root, avg_c):
    if root is None:
        return 0, 0, avg_c
    else:
        l_sum, lc, avg_c = dfs_count_sum(root.left, avg_c)
        r_sum, rc, avg_c = dfs_count_sum(root.right, avg_c)
        sub_sum = l_sum + r_sum + root.val
        count = lc + rc + 1

        if sub_sum // count == root.val:
            avg_c += 1
        print(sub_sum, count, avg_c)
    return sub_sum, count, avg_c


lst = []


def dfs_bit(root, sum):
    if root is None:
        return
    elif root.left is None and root.right is None:
        lst.append(10 * sum + root.val)
    else:
        cur = 10 * sum + root.val
        dfs_bit(root.left, cur)
        dfs_bit(root.right, cur)


num_nodes = 10
graph = defaultdict(list)
fine = [0 for _ in range(num_nodes)]


def dfs_loop(i):
    if fine[i] == 1:
        # no loop if want to take class i
        return True
    if fine[i] == -1:
        return False
    else:
        fine[i] = -1  # in this round, visited but not finished
        for j in graph[i]:
            dup = dfs_loop(j)
            if dup == False: return False
        fine[i] = 1  # never loop, can set as 1
        return True

import heapq
def prim(G, s):
    P, Q = {}, [(0, None, s)]
    while Q:
        w, p, u = heapq.heappop(Q)
        if u in P: continue # 如果目标点在生成树中，跳过
        P[u] = p # 记录目标点不在生成树中
        for v, w in G[u].items():
            heapq.heappush(Q, (w, u, v)) # 将u点的出边入堆
    return P
# T = prim(G, 1)
# sum_count = 0
# for k, v in T.items():
# if v !=None:
# sum_count += G[k][v]
#
# print(sum_count)
# print(T)
# 结果为19
# {1: None, 2: 1, 3: 1, 4: 3, 5: 6, 6: 4}

def Dijkstra(G, start):
    start = start - 1
    inf = float('inf')
    node_num = len(G)
    visited = [0] * node_num
    dis = {node: G[start][node] for node in range(node_num)}
    parents = {node: -1 for node in range(node_num)}
    visited[start] = 1
    last_point = start

    for i in range(node_num - 1):
        min_dis = inf
        for j in range(node_num):
            if visited[j] == 0 and dis[j] < min_dis:
                min_dis = dis[j]
                last_point = j
        visited[last_point] = 1
        if i == 0:
            parents[last_point] = start + 1
        for k in range(node_num):
            if G[last_point][k] < inf and dis[k] > dis[last_point] + G[last_point][k]:
                dis[k] = dis[last_point] + G[last_point][k]
                parents[k] = last_point + 1

    return {key + 1: values for key, values in dis.items()}, \
           {key + 1: values for key, values in parents.items()}


if __name__ == '__main__':
    inf = float('inf')
    G = [[0, 1, 12, inf, inf, inf],
         [inf, 0, 9, 3, inf, inf],
         [inf, inf, 0, inf, 5, inf],
         [inf, inf, 4, 0, 13, 15],
         [inf, inf, inf, inf, 0, 4],
         [inf, inf, inf, inf, inf, 0]]
    dis, parents = Dijkstra(G, 1)
    print("dis: ", dis)
    print("parents: ", parents)


class Graph:
    def __init__(self, vertices):
        self.V = vertices  # 图的顶点数
        self.edges = []    # 存储图的边

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))  # 添加边 (u, v) 和权重 w

    def bellman_ford(self, src):
        # 初始化距离数组，所有顶点的距离设为正无穷
        dist = [float("inf")] * self.V
        dist[src] = 0

        # 逐步更新距离数组
        for _ in range(self.V - 1):
            for u, v, w in self.edges:
                if dist[u] != float("inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # 检测负权重环路
        for u, v, w in self.edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                print("图中存在负权重环路")
                return

        # 打印最终的最短路径
        print("顶点 到 源点的最短距离")
        for i in range(self.V):
            print(f"{i}\t\t{dist[i]}")

# 使用示例
g = Graph(5)
g.add_edge(0, 1, -1)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 2)
g.add_edge(1, 4, 2)
g.add_edge(3, 2, 5)
g.add_edge(3, 1, 1)
g.add_edge(4, 3, -3)

g.bellman_ford(0)
