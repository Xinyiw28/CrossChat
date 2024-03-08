import pandas as pd
import numpy as np
import networkx as nx
import time
import logging
from itertools import product

def bron_kerbosch(graph, r=set(), p=None, x=set()):
    if p is None:
        p = set(graph.nodes)

    if not p and not x:
        yield r
    else:
        u = next(iter(p | x))  # Choose a pivot vertex
        for v in p - set(graph[u]):
            yield from bron_kerbosch(graph, r | {v}, p & set(graph[v]), x & set(graph[v]))
            p.remove(v)
            x.add(v)

def find_largest_complete_subgraph(graph):
    cliques = list(bron_kerbosch(graph))
    return max(cliques, key=len)

def find_all_complete_subgraph(graph):
    cliques = list(bron_kerbosch(graph))
    return cliques

def set_to_tree(s, support, inclusive_threshold):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(s))
    for n1 in list(s):
        edge_score = 0
        edge_node = n1
        for n2 in list(s):
            i_in_j = np.dot(support[n1],support[n2]) / np.sum(support[n2])
            j_in_i = np.dot(support[n1],support[n2]) / np.sum(support[n1])
            if 1 > i_in_j > edge_score and j_in_i > inclusive_threshold and j_in_i > i_in_j:
                edge_score = i_in_j
                edge_node = n2
        if edge_node != n1:
            graph.add_edge(edge_node, n1)

    return graph

def find_multiscale_trees(lr_sup_mtx, remove_cells_prop=0.95, support_size_threshold=20, inclusive_threshold=0.9, disjoint_threshold=0.99, tree_size=6, tree_scales=2):
    #load support one-hot vectors
    support = np.copy(lr_sup_mtx)
    everywhere_ind = np.where(np.sum(support/support.shape[1], axis=1) > remove_cells_prop)[0]
    for ind in everywhere_ind:
        support[ind] = np.zeros(support.shape[1])
    #generate adjacency matrix among genes with support larger than support_size_threshold

    #calculate the inclusion/disjointness relation between gene supports
    score_mtx = np.zeros((support.shape[0], support.shape[0]))
    for i in range(support.shape[0]):
        for j in range(support.shape[0]):
            u = support[i]
            v = support[j]
            if np.sum(u) > support_size_threshold and np.sum(v) > support_size_threshold:
                inclusion_score = max(np.dot(u,v) / np.sum(u), np.dot(u,v) / np.sum(v))
                #disjointness_score = max(np.dot(1 - u,v) / np.sum(v), np.dot(u,1 - v) / np.sum(u))
                disjointness_score = min(1 - np.dot(u,v)/np.sum(u), 1 - np.dot(u,v)/np.sum(v))
                score_mtx[i][j] = max(inclusion_score, disjointness_score - disjoint_threshold + inclusive_threshold)
            score_mtx[i][i] = 0

    #select the gene indices with support larger than support_size_threshold
    connected_indices_bool = np.sum(score_mtx,axis=1) > 0
    connected_indices = np.where(connected_indices_bool)[0]

    logging.info(f"there are {len(connected_indices)} connected indices out of {support.shape[0]} lr")

    #define the adjacency matrix in boolean form
    adj_mtx = np.copy(score_mtx)
    adj_mtx = adj_mtx[connected_indices_bool,:][:,connected_indices_bool]
    adj_mtx = adj_mtx >= inclusive_threshold

    #define the big graph
    Graph = nx.from_numpy_matrix(adj_mtx)

    assert len(Graph.edges()) < 16000, f"{len(Graph.edges())} edges are too many to finish on time"

    #stime = time.time()
    cliques = find_all_complete_subgraph(Graph)

    big_cliques = [clique for clique in cliques if len(clique) >= tree_size]
    assert len(big_cliques) < 3000000, f"{len(big_cliques)} complete subgraphs are too many"
    big_cliques = sorted(big_cliques, key=len, reverse=True)

    #separate real multi-scale trees from isolated genes
    list_of_trees = []
    for clique in big_cliques:
        original_indices = [connected_indices[id] for id in list(clique)]
        clique_Digraph = nx.DiGraph()
        clique_Digraph.add_nodes_from(original_indices)

        for i in original_indices:
            for j in original_indices:
                u = support[i]
                v = support[j]
                if np.dot(u,v) / np.sum(u) >= inclusive_threshold and i != j and np.sum(v) > np.sum(u):
                    clique_Digraph.add_edge(j, i)

        if len(nx.dag_longest_path(clique_Digraph)) >= tree_scales: # Remove trees with number of scales less than n_scales
            for c in nx.connected_components(clique_Digraph.to_undirected()):
                if c not in list_of_trees:
                    list_of_trees.append(c)

    sorted_trees = sorted(list_of_trees, key=len, reverse=True)
    big_trees = [tree for tree in sorted_trees if len(tree) >= tree_size]

    #remove subsets
    if len(big_trees)>0:
        unique_sorted_trees = [big_trees[0]]
        for tree in big_trees:
            subset_score = 0
            for parent_tree in unique_sorted_trees:
                subset_score += tree.issubset(parent_tree)
            if subset_score == 0:
                unique_sorted_trees.append(tree)
    else:
        return []

    logging.info(f"there are {len(unique_sorted_trees)} unique trees with size at least {tree_size}")

    final_trees = [set_to_tree(s, support, inclusive_threshold) for s in unique_sorted_trees]

    return final_trees

def interaction_btw_trees(ligand_tree, receptor_tree, LR_ind_list):
    #LR_ind_list: existing LR pairs from database in all_LR_filtered
    lr_pairs = product(list(ligand_tree.nodes), list(receptor_tree.nodes))

    return list(set(lr_pairs).intersection(set(LR_ind_list)))



