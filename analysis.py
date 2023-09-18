import math
import helpers as h
import matplotlib.pyplot as plt


print("__________________________________")


def kruskals_adj_list(G):
    AL = G.get_adj_list()
    uf = h.UnionFind()
    uf.make_set(G.m_nodes)
    edges = []
    for node in G.m_nodes:
        for edge in AL[node]:
            edges.append((node, edge[0], edge[1]))
    edges.sort(key=lambda x: x[2])
    mst = []
    for edge in edges:
        if uf.op_find(edge[0]) != uf.op_find(edge[1]):
            uf.op_union(edge[0], edge[1])
            mst.append(edge)
    return mst

def kruskals_adj_matrix(G):
    AM = G.get_adj_matrix()
    uf = h.UnionFind()
    uf.make_set(G.m_nodes)
    edges = []

    for i in range(G.m_num_of_nodes):
        for j in range(G.m_num_of_nodes):
            if AM[i][j] != math.inf:
                edges.append((i, j, AM[i][j]))
    edges.sort(key=lambda x: x[2])
    mst = []
    for edge in edges:
        if uf.op_find(edge[0]) != uf.op_find(edge[1]):
            uf.op_union(edge[0], edge[1])
            mst.append(edge)
    return mst

def prims_lazy_adj_matrix(G):
    pq = h.PriorityQueue()
    mst = []
    marked = [False for _ in range(G.m_num_of_nodes)]

    AM = G.get_adj_matrix()
    for i in range(G.m_num_of_nodes):
        if marked[i]:
            continue
        fnode = G.m_nodes[i]
        marked[fnode] = True
        # add all edges from the first node to the priority queue
        for j in range(G.m_num_of_nodes):
            if AM[fnode][j] != math.inf:
                pq.push((fnode, j), AM[fnode][j])

        while not pq.empty():
            w, edge = pq.pop()
            if marked[edge[1]]:
                continue
            mst.append((edge[0], edge[1], w))
            marked[edge[1]] = True
            for j in range(G.m_num_of_nodes):
                if AM[edge[1]][j] != math.inf:
                    pq.push((edge[1], j), AM[edge[1]][j])
    # if len(mst) != G.m_num_of_nodes - 1:
    #     print(f"Disconnected graph, {len(mst)} edges found, {G.m_num_of_nodes} nodes")
    return mst

def prims_lazy_adj_list(G):
    pq = h.PriorityQueue()
    mst = []
    marked = [False for _ in range(G.m_num_of_nodes)]

    AL = G.get_adj_list()
    for i in range(G.m_num_of_nodes):
        if marked[i]:
            continue
        fnode = G.m_nodes[i]
        marked[fnode] = True
        # add all edges from the first node to the priority queue
        for i in AL[fnode]:
            pq.push((fnode, i[0]), i[1])

        while not pq.empty():
            w, edge = pq.pop()
            if marked[edge[1]]:
                continue
            mst.append((edge[0], edge[1], w))
            marked[edge[1]] = True
            for i in AL[edge[1]]:
                pq.push((edge[1], i[0]), i[1])
    # if len(mst) != G.m_num_of_nodes - 1:
    #     print(f"Disconnected graph, {len(mst)} edges found, {G.m_num_of_nodes} nodes")
    return mst

def prims_eager_adj_list(G):
    pq = h.PriorityQueueDict()
    mst = []
    marked = [False for _ in range(G.m_num_of_nodes)]
    AL = G.get_adj_list()
    for i in range(G.m_num_of_nodes):
        if marked[i]:
            continue
        fnode = G.m_nodes[i]
        marked[fnode] = True
        # add all edges from the first node to the priority queue
        for i in AL[fnode]:
            pq.push(i[0], (fnode, i[0]), i[1])
        while not pq.empty():
            node, edge, priority = pq.pop()
            if marked[edge[1]]:
                continue
            mst.append((edge[0], edge[1], priority))
            marked[edge[1]] = True
            for i in AL[edge[1]]:
                pq.push(i[0], (edge[1], i[0]), i[1])
    # if len(mst) != G.m_num_of_nodes - 1:
    #     print(f"Disconnected graph, {len(mst)} edges found, {G.m_num_of_nodes} nodes")
    return mst

def prims_eager_adj_matrix(G):
    pq = h.PriorityQueueDict()
    mst = []
    marked = [False for _ in range(G.m_num_of_nodes)]
    AM = G.get_adj_matrix()
    for i in range(G.m_num_of_nodes):
        if marked[i]:
            continue
        fnode = G.m_nodes[i]
        marked[fnode] = True
        # add all edges from the first node to the priority queue
        for j in range(G.m_num_of_nodes):
            if AM[fnode][j] != math.inf:
                pq.push(j, (fnode, j), AM[fnode][j])
        while not pq.empty():
            node, edge, priority = pq.pop()
            if marked[edge[1]]:
                continue
            mst.append((edge[0], edge[1], priority))
            marked[edge[1]] = True
            for j in range(G.m_num_of_nodes):
                if AM[edge[1]][j] != math.inf:
                    pq.push(j, (edge[1], j), AM[edge[1]][j])
    # if len(mst) != G.m_num_of_nodes - 1:
    #     print(f"Disconnected graph, {len(mst)} edges found, {G.m_num_of_nodes} nodes")
    return mst

# check if the algorithms are correct
def test(G):
    print("------------------------------------")
    m0 = prims_lazy_adj_list(G)
    # print(m0)
    print(f"Prim's lazy adj list: {h.find_mst_weight(m0)}")

    print("------------------------------------")
    m1 = prims_lazy_adj_matrix(G)
    # print(m1)
    print(f"Prim's lazy adj matrix: {h.find_mst_weight(m1)}")

    print("------------------------------------")
    m3 = kruskals_adj_list(G)
    # print(m3)
    print(f"Kruskal's adj list: {h.find_mst_weight(m3)}")

    print("------------------------------------")
    m2 = kruskals_adj_matrix(G)
    # print(m2)
    print(f"Kruskal's adj matrix: {h.find_mst_weight(m2)}")

    print("------------------------------------")
    m4 = prims_eager_adj_list(G)
    # print(m4)
    print(f"Prim's eager adj list: {h.find_mst_weight(m4)}")

    print("------------------------------------")
    m5 = prims_eager_adj_matrix(G)
    # print(m5)
    print(f"Prim's eager adj matrix: {h.find_mst_weight(m5)}")

    print("------------------------------------")


def MSTFactory(G, algorithm, representation):
    algorithms = {
        "kruskals": {
            "adj_list": kruskals_adj_list,
            "adj_matrix": kruskals_adj_matrix
        },
        "prims_lazy": {
            "adj_list": prims_lazy_adj_list,
            "adj_matrix": prims_lazy_adj_matrix
        },
        "prims_eager": {
            "adj_list": prims_eager_adj_list,
            "adj_matrix": prims_eager_adj_matrix
        }
    }
    return algorithms[algorithm][representation](G)
# function to keep track of the time taken to run a function
def time_test(*args, func=MSTFactory, show=False):
    import time
    start = time.time()
    func(*args)
    end = time.time()
    minutes = (end - start) // 60
    seconds = (end - start) % 60
    if show:
        print(f"{args[1]} {args[2]}  ", end="\t\t")
        if minutes == 0:
            print(f"{round(seconds,4)} seconds")
        else:
            print(f"{minutes} minutes {round(seconds,4)} seconds")
    return round(seconds, 4)

def EdgeDensityVsTime(algorithms, representations, ITERATIONS=40, NODES=5000, INITIAL_EDGES=5000):
    time_taken = {}
    for algorithm in algorithms:
        time_taken[algorithm] = {}
        for representation in representations:
            time_taken[algorithm][representation] = []
    for i in range(1, ITERATIONS+1):
        print(f" Testing iteration {i}",end="\r", flush=True)
        G = h.randomGraph(NODES, INITIAL_EDGES*i, 1, 10)
        for algorithm in algorithms:
            for representation in representations:
                time_taken[algorithm][representation].append(time_test(G, algorithm, representation))
    print(f" Testing Complete         ")
    for algorithm in algorithms:
        for representation in representations:
            plt.plot([(i*INITIAL_EDGES)//1000 for i in range(1,ITERATIONS+1)],time_taken[algorithm][representation], label=f"{algorithm}_{representation}")
    
    plt.xlabel("Edges in thousands")
    plt.ylabel("Time taken in seconds")
    plt.legend()
    plt.show()

def NodeDensityVsTime(algorithms, representations, ITERATIONS=30, INITIAL_NODES=5000, EDGES=600000):
    time_taken = {}
    for algorithm in algorithms:
        time_taken[algorithm] = {}
        for representation in representations:
            time_taken[algorithm][representation] = []
    for i in range(1, ITERATIONS+1):
        print(f" Testing iteration {i}",end="\r", flush=True)
        G = h.randomGraph(INITIAL_NODES*i, EDGES, 1, 10)
        for algorithm in algorithms:
            for representation in representations:
                time_taken[algorithm][representation].append(time_test(G, algorithm, representation))
    print(f" Testing Complete         ")
    for algorithm in algorithms:
        for representation in representations:
            plt.plot([INITIAL_NODES*i//1000 for i in range(1,ITERATIONS+1)],time_taken[algorithm][representation], label=f"{algorithm}_{representation}")
    
    plt.xlabel("Nodes in thousands")
    plt.ylabel("Time taken in seconds")
    plt.legend()
    plt.show()


NodeDensityVsTime(["kruskals", "prims_lazy", "prims_eager"], ["adj_list"], ITERATIONS=30, INITIAL_NODES=5000, EDGES=600000)
EdgeDensityVsTime(["kruskals","prims_lazy","prims_eager"],["adj_list"], ITERATIONS=40, NODES=5000, INITIAL_EDGES=5000)
NodeDensityVsTime(["kruskals","prims_lazy","prims_eager"],["adj_matrix"], ITERATIONS=10, INITIAL_NODES=500, EDGES=60000)
EdgeDensityVsTime(["kruskals","prims_lazy","prims_eager"],["adj_matrix"], ITERATIONS=20, NODES=5000, INITIAL_EDGES=5000)

EdgeDensityVsTime(["kruskals","prims_lazy","prims_eager"],["adj_list","adj_matrix"], ITERATIONS=30, NODES=1000, INITIAL_EDGES=6000)
NodeDensityVsTime(["kruskals","prims_lazy","prims_eager"],["adj_list","adj_matrix"], ITERATIONS=10, INITIAL_NODES=1000, EDGES=60000)
print("------------------------------------")