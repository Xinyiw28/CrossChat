import pandas as pd
import numpy as np
import umap
import scanpy as sc
from sklearn.decomposition import PCA
import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from .Data_preparation import prepare_lr_sup_mtx,prepare_gene_exp_dict,load_files,filter_all_LR,multiscale_clustering,\
compute_genes_geo_avg_vector,get_wilcox_score,get_onehot_ls,get_CCC_mtx,get_lr_exp_in_clusters,cluster_exp_ls, \
select_partitions,draw_multiscale_umap,get_pathway_genes,get_Markov_time_ls,get_cluster_size_ls,prepare_lr_union_sup_mtx,\
compute_interaction,get_lr_frequency_in_trees,get_new_lr_sup
from .Multiple_trees import find_multiscale_trees, interaction_btw_trees
from .Visualization import get_node_level_top_down,draw_MMT,draw_CCC_between_LR_union_MMT,draw_CCC_between_MMT

class CrossChatT:

    def __init__(self, adata, species="human"):
        """
        Initialize the CrossChatT object. 
        
        :params: adata is the input adata object 
        :params: species is either human or mouse 
        :return:  the CrossChatT object
        """     
        self.adata = adata
        self.mtx = adata.X
        self.ncells = self.mtx.shape[0]
        self.genenames = pd.DataFrame(adata.var_names.map(lambda x:x.upper()))
        self.species = species
        self.all_LR, self.cofactor_input, self.complex_input = load_files(species=self.species)

    def prepare_adata(self, normalize=False, scale=False, input='allgenes'):
        """
        Prepares the CrossChatT object. 
        
        :params: normalize is True if data needs to be normalized
        :params: scale is True if data needs to be scaled
        :params: input is allgenes if use all genes 
        :return:  the CrossChatH object
        """
        if normalize == True:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
        if input=='allgenes':
            sc.pp.highly_variable_genes(self.adata, n_top_genes=2000)
        if scale == True:
            sc.pp.scale(self.adata, max_value=10)
        sc.tl.pca(self.adata, svd_solver='arpack')
        sc.pp.neighbors(self.adata)
        sc.tl.umap(self.adata)
        _, _, L_ls, separate_L = prepare_lr_sup_mtx(self.all_LR, self.complex_input, 'L', self.mtx.T, self.genenames)
        _, _, R_ls, separate_R = prepare_lr_sup_mtx(self.all_LR, self.complex_input, 'R', self.mtx.T, self.genenames)
        self.ligand_exp_dict, self.receptor_exp_dict, self.cofactor_exp_dict = prepare_gene_exp_dict(L_ls, R_ls, separate_L,
                                                                                      separate_R, self.cofactor_input,
                                                                                      self.genenames, self.mtx.T)
        self.all_LR_filtered = filter_all_LR(self.all_LR, self.ligand_exp_dict, self.receptor_exp_dict, self.ncells, threshold=0.005)

    def Draw_annotations_umap(self):
        """
        Draw umap of cell type annotations

        :return:  multiscale umap 
        """           
        sc.pl.umap(self.adata, color="annotations", save=False)

    def Binarization(self,threshold=0):
        #Input to Binarization is normalized but not scaled gene expression matrix
        #Threshold each gene on all cells, throw out the cells with lowest gene expression according to threshold
        ncells = self.adata.X.shape[0]
        matrix = self.adata.X
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i).toarray()[0]
            positive_indices = row > 0
            n_remove = int(positive_indices.sum()*threshold)  #number of elements to change to 0
            if np.count_nonzero(positive_indices) > n_remove:
                positive_values = row[positive_indices]
                smallest_n_indices = np.argsort(positive_values)[:n_remove]
                actual_indices = np.nonzero(positive_indices)[0][smallest_n_indices]
                matrix[i, actual_indices] = 0
        matrix = matrix.tocsr()
        self.adata.X = matrix

    def Detect_trees(self,type="l",remove_cells_prop=0.9,support_size_threshold=30, inclusive_threshold=0.9,
                     disjoint_threshold=0.95,tree_size=4,tree_scales=3):
        """
        Detect trees involved in CCC in scRNA-seq data.

        :params: type is either "l" (ligand) or "r" (receptor)
        :params: remove_cells_prop keeps genes that are present in more than the proportion of cells
        :params: support_size_threshold keeps genes that are present in more than support_size_threshold cells
        :params: tree_size is the threshold of number of nodes in the tree
        :params: tree_scales is the number of levels in the tree
        :return:  the CrossChatT object with detected trees
        """            

        #type is l,r, or lr
        self.L_sup_mtx, all_LR_L_ind, self.L_ls, separate_L = prepare_lr_sup_mtx(self.all_LR_filtered, self.complex_input, 'L',
                                                                       self.mtx.T, self.genenames)
        self.R_sup_mtx, all_LR_R_ind, self.R_ls, separate_R = prepare_lr_sup_mtx(self.all_LR_filtered, self.complex_input, 'R',
                                                                       self.mtx.T, self.genenames)
        if type == "l":
            trees = find_multiscale_trees(self.L_sup_mtx, remove_cells_prop=remove_cells_prop,
                                          support_size_threshold=support_size_threshold,
                                          inclusive_threshold=inclusive_threshold,
                                          disjoint_threshold=disjoint_threshold, tree_size=tree_size,
                                          tree_scales=tree_scales)
            self.ligand_trees = trees
        elif type == "r":
            trees = find_multiscale_trees(self.R_sup_mtx, remove_cells_prop=remove_cells_prop,
                                          support_size_threshold=support_size_threshold,
                                          inclusive_threshold=inclusive_threshold,
                                          disjoint_threshold=disjoint_threshold, tree_size=tree_size,
                                          tree_scales=tree_scales)
            self.receptor_trees = trees
        elif type == "lr_union":
            self.LR_sup_mtx = prepare_lr_union_sup_mtx(self.all_LR_filtered, self.L_sup_mtx, self.R_sup_mtx, self.L_ls, self.R_ls)
            trees = find_multiscale_trees(self.LR_sup_mtx, remove_cells_prop=remove_cells_prop,
                                          support_size_threshold=support_size_threshold,
                                          inclusive_threshold=inclusive_threshold,
                                          disjoint_threshold=disjoint_threshold, tree_size=tree_size,
                                          tree_scales=tree_scales)
            self.lr_trees = trees

    def Detect_trees_S(self,spatial_range=300,type="l",remove_cells_prop=0.9,support_size_threshold=30, inclusive_threshold=0.9,
                     disjoint_threshold=0.95,tree_size=4,tree_scales=3):
        """
        Detect trees involved in CCC in spatial data.

        :params: type is either "l" (ligand) or "r" (receptor)
        :params: remove_cells_prop keeps genes that are present in more than the proportion of cells
        :params: support_size_threshold keeps genes that are present in more than support_size_threshold cells
        :params: tree_size is the threshold of number of nodes in the tree
        :params: tree_scales is the number of levels in the tree
        :return:  the CrossChatT object with detected trees
        """    
        #type is l,r, or lr
        self.L_sup_mtx, all_LR_L_ind, self.L_ls, separate_L = prepare_lr_sup_mtx(self.all_LR_filtered, self.complex_input, 'L',
                                                                       self.mtx.T, self.genenames)
        self.R_sup_mtx, all_LR_R_ind, self.R_ls, separate_R = prepare_lr_sup_mtx(self.all_LR_filtered, self.complex_input, 'R',
                                                                       self.mtx.T, self.genenames)
        ligands_ls = []
        for receptor in self.R_ls:
            ligand_ls = []
            for i in range(len(self.all_LR_filtered)):
                if self.all_LR_filtered.iloc[i]['Receptor'] == receptor:
                    ligand = self.all_LR_filtered.iloc[i]['Ligand']
                    ligand_ls.append(ligand)
            ligand_ls = list(set(ligand_ls))
            ligands_ls.append(ligand_ls)

        receptors_ls = []
        for ligand in self.L_ls:
            receptor_ls = []
            for i in range(len(self.all_LR_filtered)):
                if self.all_LR_filtered.iloc[i]['Ligand'] == ligand:
                    receptor = self.all_LR_filtered.iloc[i]['Receptor']
                    receptor_ls.append(receptor)
            receptor_ls = list(set(receptor_ls))
            receptors_ls.append(receptor_ls)

        spatial_dist_mtx = distance_matrix(self.adata.obsm['spatial'], self.adata.obsm['spatial'])
        spatial_dist_mtx[spatial_dist_mtx < spatial_range] = 1
        spatial_dist_mtx[spatial_dist_mtx > spatial_range] = 0

        new_L_sup_mtx = np.zeros_like(self.L_sup_mtx)
        for i, ligand in enumerate(self.L_ls):
            ligand_sup = self.L_sup_mtx[i]
            new_ligand_sup = np.zeros_like(ligand_sup)
            receptors = receptors_ls[i]
            for receptor in receptors:
                receptor_sup = np.prod(self.receptor_exp_dict[receptor], axis=0)
                receptor_sup[receptor_sup > 0] = 1
                new_ligand_sup += np.squeeze(get_new_lr_sup(ligand_sup, receptor_sup, spatial_dist_mtx)[0].T)
            new_ligand_sup[new_ligand_sup > 0] = 1
            new_L_sup_mtx[i] = new_ligand_sup

        new_R_sup_mtx = np.zeros_like(self.R_sup_mtx)
        for i, receptor in enumerate(self.R_ls):
            receptor_sup = self.R_sup_mtx[i]
            new_receptor_sup = np.zeros_like(receptor_sup)
            ligands = ligands_ls[i]
            for ligand in ligands:
                ligand_sup = np.prod(self.ligand_exp_dict[ligand], axis=0)
                ligand_sup[ligand_sup > 0] = 1
                new_receptor_sup += np.squeeze(get_new_lr_sup(receptor_sup, ligand_sup, spatial_dist_mtx)[0].T)
            new_receptor_sup[new_receptor_sup > 0] = 1
            new_R_sup_mtx[i] = new_receptor_sup

        self.spatial_L_sup_mtx = new_L_sup_mtx
        self.spatial_R_sup_mtx = new_R_sup_mtx

        if type == "l":
            trees = find_multiscale_trees(self.spatial_L_sup_mtx, remove_cells_prop=remove_cells_prop,
                                          support_size_threshold=support_size_threshold,
                                          inclusive_threshold=inclusive_threshold,
                                          disjoint_threshold=disjoint_threshold, tree_size=tree_size,
                                          tree_scales=tree_scales)
            self.ligand_trees = sorted(trees, key=lambda x: x.number_of_nodes(), reverse=True)
        elif type == "r":
            trees = find_multiscale_trees(self.spatial_R_sup_mtx, remove_cells_prop=remove_cells_prop,
                                          support_size_threshold=support_size_threshold,
                                          inclusive_threshold=inclusive_threshold,
                                          disjoint_threshold=disjoint_threshold, tree_size=tree_size,
                                          tree_scales=tree_scales)
            self.receptor_trees = sorted(trees, key=lambda x: x.number_of_nodes(), reverse=True)
        elif type == "lr_union":
            self.spatial_LR_sup_mtx = prepare_lr_union_sup_mtx(self.all_LR_filtered, self.spatial_L_sup_mtx, self.spatial_R_sup_mtx, self.L_ls, self.R_ls)
            trees = find_multiscale_trees(self.spatial_LR_sup_mtx, remove_cells_prop=remove_cells_prop,
                                          support_size_threshold=support_size_threshold,
                                          inclusive_threshold=inclusive_threshold,
                                          disjoint_threshold=disjoint_threshold, tree_size=tree_size,
                                          tree_scales=tree_scales)
            self.lr_trees = sorted(trees, key=lambda x: x.number_of_nodes(), reverse=True)

    def Draw_MMT(self,type="l",tree_inds=None,nodesize=20):
        """
        Visualize detected trees

        :params: type is either "l" or "r"
        :params: tree_inds is the index of tree in all detected trees
        :return:  the visualization of detected trees
        """ 
        # type is l,r
        if type == "l":
            trees = self.ligand_trees
            sup_mtx = self.L_sup_mtx
            lr_ls = self.L_ls
        if type == "r":
            trees = self.receptor_trees
            sup_mtx = self.R_sup_mtx
            lr_ls = self.R_ls
        for tree_ind in tree_inds:
            print(tree_ind)
            node_pos = nx.nx_pydot.graphviz_layout(trees[tree_ind], prog='dot')
            node_level_dict = get_node_level_top_down(trees[tree_ind])
            tree_node_onehot_ls = []
            for level in range(len(node_level_dict)):
                tree_node_onehot_ls.append(sup_mtx[node_level_dict[level]])
            node_ls = sum(node_level_dict.values(), [])
            node_mapping_dict = dict()
            for ind, node in enumerate(node_ls):
                node_mapping_dict[node] = ind
            size_ls = get_cluster_size_ls(tree_node_onehot_ls)
            draw_MMT(trees[tree_ind], node_pos, size_ls, lr_ls=lr_ls,
                     lr_node_mapping_dict=node_mapping_dict, nodesize=nodesize)

    def Draw_MMT_lr_union(self,tree_inds=None):
        """
        Visualize detected trees with lr_union as input 

        :params: tree_inds is the index of tree in all detected trees
        :return:  the visualization of detected trees
        """ 
        for tree_ind in tree_inds:
            print(tree_ind)
            LR_union_tree_node_ls = list(self.lr_trees[tree_ind].nodes)
            LR_union_tree = self.lr_trees[tree_ind]
            LR_union_node_pos = nx.nx_pydot.graphviz_layout(self.lr_trees[tree_ind], prog='dot')
            LR_union_node_mapping_dict = dict()
            for i, node in enumerate(LR_union_tree_node_ls):
                LR_union_node_mapping_dict[node] = i
            tree_ligand_onehot_ls = []
            tree_receptor_onehot_ls = []
            CCC_vector = []
            tree_ligand_ind_ls = []
            tree_receptor_ind_ls = []
            for node in LR_union_tree_node_ls:
                L_ind = np.where(self.L_ls == self.all_LR_filtered.iloc[node]['Ligand'])[0][0]
                R_ind = np.where(self.R_ls == self.all_LR_filtered.iloc[node]['Receptor'])[0][0]
                tree_ligand_ind_ls.append(L_ind)
                tree_receptor_ind_ls.append(R_ind)
                tree_ligand_onehot_ls.append(self.L_sup_mtx[L_ind])
                tree_receptor_onehot_ls.append(self.R_sup_mtx[R_ind])
                ligands_ls = [self.all_LR_filtered.iloc[node]['Ligand']]
                receptors_ls = [self.all_LR_filtered.iloc[node]['Receptor']]
                CCC = compute_interaction(node, self.L_sup_mtx[L_ind], self.R_sup_mtx[R_ind], self.all_LR_filtered, self.ligand_exp_dict,
                                          self.receptor_exp_dict, self.cofactor_exp_dict)
                if CCC > 0.3:
                    CCC_vector.append(CCC)
                else:
                    CCC_vector.append(0)

            L_size_ls = np.sum(tree_ligand_onehot_ls, axis=1) / self.ncells
            R_size_ls = np.sum(tree_receptor_onehot_ls, axis=1) / self.ncells
            L_color_ls = []
            R_color_ls = []
            for node in tree_ligand_ind_ls:
                L_color_ls.append(np.where(list(set(tree_ligand_ind_ls)) == node)[0][0] % 10)
            for node in tree_receptor_ind_ls:
                R_color_ls.append(np.where(list(set(tree_receptor_ind_ls)) == node)[0][0] % 10)
            draw_CCC_between_LR_union_MMT(LR_union_tree, LR_union_node_pos, self.all_LR_filtered, L_size_ls, R_size_ls,
                                          L_color_ls, R_color_ls, CCC_vector,
                                          LR_union_node_mapping_dict=LR_union_node_mapping_dict, CCC_nodesize=2,
                                          save=None)

    def Draw_CCC_between_MMT(self,lr_tree_inds):
        """
        Visualize CCC detected between trees

        :params: lr_tree_inds is the index of ligand/receptor trees
        :return:  the visualization of CCC between detected trees
        """ 
        LR_list = self.all_LR_filtered[['Ligand', 'Receptor']].values
        LR_ind_list = list(map(lambda x: (np.where(self.L_ls == x[0])[0][0], np.where(self.R_ls == x[1])[0][0]), LR_list))

        for lr_tree_ind in lr_tree_inds:
            L_tree_num = lr_tree_ind[0]
            R_tree_num = lr_tree_ind[1]
            
            L_node_pos = nx.nx_pydot.graphviz_layout(self.ligand_trees[L_tree_num], prog='dot')
            L_node_level_dict = get_node_level_top_down(self.ligand_trees[L_tree_num])
            R_node_pos = nx.nx_pydot.graphviz_layout(self.receptor_trees[R_tree_num], prog='dot')
            R_node_level_dict = get_node_level_top_down(self.receptor_trees[R_tree_num])

            tree_ligand_onehot_ls = []
            for level in range(len(L_node_level_dict)):
                tree_ligand_onehot_ls.append(self.L_sup_mtx[L_node_level_dict[level]])
            tree_receptor_onehot_ls = []
            for level in range(len(R_node_level_dict)):
                tree_receptor_onehot_ls.append(self.R_sup_mtx[R_node_level_dict[level]])
    
            L_node_ls = sum(L_node_level_dict.values(), [])
            L_node_mapping_dict = dict()
            R_node_ls = sum(R_node_level_dict.values(), [])
            R_node_mapping_dict = dict()
    
            for ind, node in enumerate(L_node_ls):
                L_node_mapping_dict[node] = ind
            for ind, node in enumerate(R_node_ls):
                R_node_mapping_dict[node] = ind
    
            L_size_ls = get_cluster_size_ls(tree_ligand_onehot_ls)
            R_size_ls = get_cluster_size_ls(tree_receptor_onehot_ls)
            ligands_ls = list(map(lambda x: x.upper(), self.L_ls[self.ligand_trees[L_tree_num].nodes]))
            receptors_ls = list(map(lambda x: x.upper(), self.R_ls[self.receptor_trees[R_tree_num].nodes]))
    
            CCC = get_CCC_mtx(self.all_LR_filtered, ligands_ls, receptors_ls, tree_ligand_onehot_ls,
                              tree_receptor_onehot_ls, self.ligand_exp_dict, self.receptor_exp_dict,
                              self.cofactor_exp_dict, CCC_threshold=0)
    
            CCC_mtx = np.concatenate(
                [np.concatenate([CCC[(l_level, r_level)] for l_level in range(len(tree_ligand_onehot_ls))], axis=0) for
                 r_level in range(len(tree_receptor_onehot_ls))], axis=1)
            lr_pairs_inds = interaction_btw_trees(self.ligand_trees[L_tree_num], self.receptor_trees[R_tree_num],
                                                  LR_ind_list)

            draw_CCC_between_MMT(self.ligand_trees[L_tree_num], L_node_pos, self.receptor_trees[R_tree_num], R_node_pos, self.L_ls,
                                 L_size_ls, self.R_ls, R_size_ls, CCC_mtx, lr_pairs_inds,
                                 L_node_mapping_dict=L_node_mapping_dict, R_node_mapping_dict=R_node_mapping_dict,
                                 save=None)

    def find_interacting_trees(self, num_ligand_trees=10, num_receptor_trees=10, num_interation_threshold=1):
        """
        Obtain the pairs of interacting trees

        :params: num_ligand_trees is the number of top detected ligand trees that user is interested
        :params: num_receptor_trees is the number of top detected receptor trees that user is interested
        :params: number_interaction_threshold is the threshold of CCC strength 
        :return:  the list of interacting trees
        """         
        
        interacting_tree_inds = []
        LR_list = self.all_LR_filtered[['Ligand', 'Receptor']].values
        LR_ind_list = list(map(lambda x: (np.where(self.L_ls == x[0])[0][0], np.where(self.R_ls == x[1])[0][0]), LR_list))
        for l in range(num_ligand_trees):
            ligand_tree = self.ligand_trees[l]
            for r in range(num_receptor_trees):
                receptor_tree = self.receptor_trees[r]
                n_interactions = len(interaction_btw_trees(ligand_tree, receptor_tree, LR_ind_list))
                if n_interactions >= num_interation_threshold:
                    interacting_tree_inds.append([l, r, n_interactions])

        return sorted(interacting_tree_inds, key=lambda x:x[2], reverse=True)

    def plot_lr_frequency(self,type='l'):
        """
        Plot frequency of ligand/receptor occurrence in ligands/receptors trees

        :params: type is the type of ligands or receptors
        :return:  the visualization of frequency of ligand/receptor occurrence in ligands/receptors trees
        """         
        orig_L_indices_in_trees = []
        orig_R_indices_in_trees = []
        for tree in self.ligand_trees:
            for node in tree.nodes:
                orig_L_indices_in_trees.append(node)
        for tree in self.receptor_trees:
            for node in tree.nodes:
                orig_R_indices_in_trees.append(node)
        L_indices_in_trees = list(set(orig_L_indices_in_trees))
        R_indices_in_trees = list(set(orig_R_indices_in_trees))
        L_freq_dict = get_lr_frequency_in_trees(orig_L_indices_in_trees, self.L_ls)
        R_freq_dict = get_lr_frequency_in_trees(orig_R_indices_in_trees, self.R_ls)
        if type == 'l':
            plt.barh(list(L_freq_dict.keys())[0:15][::-1], list(L_freq_dict.values())[0:15][::-1], color='#3C93C2')
        elif type == 'r':
            plt.barh(list(R_freq_dict.keys())[0:15][::-1], list(R_freq_dict.values())[0:15][::-1], color='#3C93C2')

    def Draw_big_tree(self,type='l',tree_inds=None):
        """
        Draw union of trees

        :params: type is the type of ligands or receptors
        :params: tree_inds is the list of indices for the set of trees user is interested in 
        :return:  the visualization of frequency of ligand/receptor occurrence in ligands/receptors trees
        """              
        selected_lr = []
        if type == 'l':
            selected_ligands = []
            for tree_ind in tree_inds:
                tree = self.ligand_trees[tree_ind]
                selected_lr.extend(list(tree.nodes()))

            selected_lr = set(selected_lr)
            big_tree = nx.DiGraph()
            big_tree.add_nodes_from(selected_lr)
            big_tree_edges = []
            for tree_ind in tree_inds:
                tree = self.ligand_trees[tree_ind]
                for edge in tree.edges:
                    big_tree_edges.append(edge)
            big_tree.add_edges_from(list(set(big_tree_edges)))

            node_pos = nx.nx_pydot.graphviz_layout(big_tree, prog='dot')
            node_level_dict = get_node_level_top_down(big_tree)
            tree_lr_onehot_ls = []
            for level in range(len(node_level_dict)):
                tree_lr_onehot_ls.append(self.L_sup_mtx[node_level_dict[level]])
            node_ls = sum(node_level_dict.values(), [])
            node_mapping_dict = dict()
            for ind, node in enumerate(node_ls):
                node_mapping_dict[node] = ind
            size_ls = get_cluster_size_ls(tree_lr_onehot_ls)

            draw_MMT(big_tree, node_pos, size_ls, lr_ls=self.L_ls, lr_node_mapping_dict=node_mapping_dict,
                     nodesize=3)



