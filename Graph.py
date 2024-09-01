from collections import defaultdict


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
        if u in P: continue  # 如果目标点在生成树中，跳过
        P[u] = p  # 记录目标点不在生成树中
        for v, w in G[u].items():
            heapq.heappush(Q, (w, u, v))  # 将u点的出边入堆
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

import heapq

# 到一个节点的最短路径必然会经过比它离起点更近的节点，而如果一个节点的当前距离值比任何剩余节点都小，那么当前的距离值一定是最小的。
def dijkstra(self, start_vertex_data):
    start_vertex = self.vertex_data.index(start_vertex_data)
    distances = [float('inf')] * self.size
    distances[start_vertex] = 0
    visited = [False] * self.size

    for _ in range(self.size):
        min_distance = float('inf')
        u = None
        for i in range(self.size):
            if not visited[i] and distances[i] < min_distance:
                min_distance = distances[i]
                u = i
        if u is None:
            break
        visited[u] = True

        for v in range(self.size):
            if self.adj_matrix[u][v] != 0 and not visited[v]:
                alt = distances[u] + self.adj_matrix[u][v]
                if alt < distances[v]:
                    distances[v] = alt
    return distances

def bellman_ford(self, start_vertex_data):
    start_vertex = self.vertex_data.index(start_vertex_data)
    distances = [float('inf')] * self.size
    distances[start_vertex] = 0

    for i in range(self.size - 1):
        for u in range(self.size):
            for v in range(self.size):
                if self.adj_matrix[u][v] != 0:
                    if distances[u] + self.adj_matrix[u][v] < distances[v]:
                        distances[v] = distances[u] + self.adj_matrix[u][v]
                        print(f"Relaxing edge {self.vertex_data[u]}-{self.vertex_data[v]}, "
                              f"Updated distance to {self.vertex_data[v]}: {distances[v]}")

    return distances


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))  # 添加边 (u, v) 和权重 w

    def bellman_ford(self, src):
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

# def Floyd(e):
#     for all [i,j]: // initialize
#         d[i,j] = e[i,j]
#     for k = 1 ~ n: // relax for k times:
#         for all [i,j]:
#             d[i,j] = min(d[i,j], d[i,k] + e[k,j])
