import heapq
import pqdict
import random


class Graph:
    def __init__(self, num_of_nodes, num_of_edges, directed=False):
        self.m_num_of_nodes = num_of_nodes
        self.m_nodes = range(self.m_num_of_nodes)
        self.m_num_of_edges = num_of_edges

        # Define the type of a graph
        self.m_directed = directed
        self.m_adj_list = {node: set() for node in self.m_nodes}

    def add_edge(self, node1, node2, weight=1):
        self.m_adj_list[node1].add((node2, weight))

        if not self.m_directed:
            self.m_adj_list[node2].add((node1, weight))
    def check_edge(self, node1, node2):
        if node2 in self.m_adj_list[node1]:
            return True
        else:
            return False
    def print_adj_list(self):
        for key in self.m_adj_list.keys():
            print("node", key, ": ", self.m_adj_list[key])

    def get_adj_list(self):
        return self.m_adj_list

    def get_adj_matrix(self):
        # initialize the adjacency matrix with infinity
        max_weight = float('inf')
        adj_matrix = [[max_weight for _ in range(
            self.m_num_of_nodes)] for _ in range(self.m_num_of_nodes)]

        # if there are multiple edges between two nodes, we take the minimum weight
        for node, edges in self.m_adj_list.items():
            for edge in edges:
                adj_matrix[node][edge[0]] = min(
                    adj_matrix[node][edge[0]], edge[1])

                if not self.m_directed:
                    adj_matrix[edge[0]][node] = min(
                        adj_matrix[edge[0]][node], edge[1])

        return adj_matrix


# add a random graph generator
def randomGraph(num_of_nodes, num_of_edges, min_weight=50, max_weight=100, seed=16323467893243212243254, directed=False):
    random.seed(seed)
    g = Graph(num_of_nodes, num_of_edges, directed=directed)

    max_edges = num_of_nodes*(num_of_nodes-1) if directed else num_of_nodes*(num_of_nodes-1)//2
    if num_of_edges > max_edges:
        return "Error: Too many edges. Maximum number of edges for this graph is " + str(max_edges)

    edge_count = 0
    while edge_count < num_of_edges:
        u = random.randint(0, num_of_nodes - 1)
        v = random.randint(0, num_of_nodes - 1)

        # check if u is connected to the entire graph
        while len(g.get_adj_list()[u]) == num_of_nodes - 1:
            u = random.randint(0, num_of_nodes - 1)

        while u == v or g.check_edge(u, v):
            v = random.randint(0, num_of_nodes - 1)

        weight = random.randint(min_weight, max_weight)
        g.add_edge(u, v, weight=weight)
        
        edge_count += 1

    return g



class UnionFind:
    parent_node = {}
    rank = {}

    def make_set(self, u):
        for i in u:
            self.parent_node[i] = i
            self.rank[i] = 0

    def op_find(self, k):
        if self.parent_node[k] != k:
            self.parent_node[k] = self.op_find(self.parent_node[k])
        return self.parent_node[k]

    def op_union(self, a, b):
        x = self.op_find(a)
        y = self.op_find(b)

        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent_node[y] = x
        elif self.rank[x] < self.rank[y]:
            self.parent_node[x] = y
        else:
            self.parent_node[x] = y
            self.rank[y] = self.rank[y] + 1

# https://www.delftstack.com/howto/python/union-find-in-python/


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def push(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        priority, item = heapq.heappop(self.elements)
        return priority, item

    def empty(self):
        return len(self.elements) == 0

    def __repr__(self):
        return str(self.elements)


class PriorityQueueDict:
    def __init__(self):
        self.elements = pqdict.pqdict()
        self.edges = {}

    def push(self, node, edge, priority):
        #check if node is already in the queue
        if node not in self.elements:
            self.elements.additem(node, priority)
            self.edges[node] = edge
        else:
            self.relax(node, edge, priority)

    def pop(self):
        key, value = self.elements.popitem()
        edge = self.edges[key]
        del self.edges[key]
        return (key, edge, value)

    def empty(self):
        return len(self.elements) == 0

    def update(self, node, edge, priority):
        self.elements.updateitem(node, priority)
        self.edges[node] = edge

    def relax(self, node, edge, priority):
        if priority < self.elements[node]:
            self.update(node, edge, priority)

    def __repr__(self):
        return [(key, self.edges[key], value) for key, value in self.elements.items()].__repr__()


def find_mst_weight(mst):
    weight = 0
    for edge in mst:
        weight += edge[2]
    return weight
