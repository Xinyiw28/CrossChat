import pandas as pd
import numpy as np
import umap
import scanpy as sc
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from .Data_preparation import prepare_lr_sup_mtx,prepare_gene_exp_dict,load_files,filter_all_LR,multiscale_clustering,\
compute_genes_geo_avg_vector,get_wilcox_score,get_onehot_ls,get_CCC_mtx,get_lr_exp_in_clusters,cluster_exp_ls, \
select_partitions,draw_multiscale_umap,get_pathway_genes,get_Markov_time_ls,obtain_spatial_pca,jaccard_dist, get_cluster_results
from .Visualization import draw_CCC

class CrossChatH:

    def __init__(self, adata, species="human", user_comm_ids=None):
        """
        Initialize the CrossChatH object. 
        
        :param adata is the input adata object 
        :param species is either human or mouse 
        :param user_comm_ids is the multiscale clustering results if user wishes to input 
        :return: the CrossChatH object
        """
        self.adata = adata
        self.mtx = adata.X
        self.ncells = self.mtx.shape[0]
        self.genenames = pd.DataFrame(adata.var_names.map(lambda x:x.upper()))
        self.L_allresults = None
        self.R_allresults = None
        self.species = species
        self.all_LR, self.cofactor_input, self.complex_input = load_files(species=self.species)
        if user_comm_ids is not None:
            self.L_allresults = get_cluster_results(user_comm_ids)
            self.R_allresults = get_cluster_results(user_comm_ids)


    def prepare_adata(self, normalize=False, scale=False, input='allgenes'):
        """
        Prepares the CrossChatH object. 
        
        :param normalize is True if data needs to be normalized
        :param scale is True if data needs to be scaled
        :param input is allgenes if use all genes 
        :return: the CrossChatH object
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
        _, _, self.L_ls, _ = prepare_lr_sup_mtx(self.all_LR_filtered, self.complex_input, 'L', self.mtx.T, self.genenames)
        _, _, self.R_ls,_ = prepare_lr_sup_mtx(self.all_LR_filtered, self.complex_input, 'R', self.mtx.T, self.genenames)

    def Multsicale_clustering(self,cluster_by="allgenes",k=15):
        """
        Runs multiscale clustering on cells based on either allgenes, or ligands, or receptors
        
        :param cluster_by is the genes that user want to use to cluster, either "allgenes" or "lr" 
        :param k: value of k in knn
        :return CrossChatH object after multiscale clustering
        """
        if cluster_by == "allgenes":
            pca_embedding = self.adata.obsm['X_pca']
            self.L_allresults = multiscale_clustering(pca_embedding, min_scale=-1, max_scale=4, n_scale=100, n_tries=20, k=k)
            self.R_allresults = self.L_allresults
        elif cluster_by == "lr":
            L_mtx = np.concatenate([self.ligand_exp_dict[self.L_ls[i]] for i in range(len(self.L_ls))], axis=0).T
            pca = PCA(n_components=50)
            L_pca = pca.fit_transform(L_mtx)
            self.L_allresults = multiscale_clustering(L_pca, min_scale=-1, max_scale=4, n_scale=100, n_tries=20, k=k)
            R_mtx = np.concatenate([self.receptor_exp_dict[self.R_ls[i]] for i in range(len(self.R_ls))], axis=0).T
            pca = PCA(n_components=50)
            R_pca = pca.fit_transform(R_mtx)
            self.R_allresults = multiscale_clustering(R_pca, min_scale=-1, max_scale=4, n_scale=100, n_tries=20, k=k)

    def Multsicale_clustering_spatial(self,cluster_by="allgenes",k=15,w=0.5):
        """
        Runs multiscale clustering on spatial data based on either allgenes, or ligands, or receptors
        
        :param cluster_by is the genes that user want to use to cluster, either "allgenes" or "lr" 
        :param k is value of k in knn 
        :param w is the weight 
        :return CrossChatH object after multiscale clustering
        """
        if cluster_by == "allgenes":
            pca_embedding = self.adata.obsm['X_pca']
            spatial_pca = obtain_spatial_pca(pca_embedding, self.adata.obsm['spatial'], w=w)
            self.L_allresults = multiscale_clustering(spatial_pca, min_scale=-1, max_scale=4, n_scale=100, n_tries=20,k=k)
            self.R_allresults = self.L_allresults
        elif cluster_by == "lr":
            L_mtx = np.concatenate([self.ligand_exp_dict[self.L_ls[i]] for i in range(len(self.L_ls))], axis=0).T
            pca = PCA(n_components=50)
            L_pca = pca.fit_transform(L_mtx)
            R_mtx = np.concatenate([self.receptor_exp_dict[self.R_ls[i]] for i in range(len(self.R_ls))], axis=0).T
            pca = PCA(n_components=50)
            R_pca = pca.fit_transform(R_mtx)
            L_spatial_pca = obtain_spatial_pca(L_pca, self.adata.obsm['spatial'], w=w)
            R_spatial_pca = obtain_spatial_pca(R_pca, self.adata.obsm['spatial'], w=w)
            self.L_allresults = multiscale_clustering(L_spatial_pca, min_scale=-1, max_scale=4, n_scale=100, n_tries=20,k=k)
            self.R_allresults = multiscale_clustering(R_spatial_pca, min_scale=-1, max_scale=4, n_scale=100, n_tries=20,k=k)

    def select_partitions(self, max_nvi=0.1, window_size=15, basin_radius=15, lr = "L"):
        """
        Select the desired hierarchical clustering.

        :param lr is "L" (ligand) or "R" (receptor)
        :return CrossChatH object after selecting partitions 
        """
        if lr == "L":
            selected_partitions, selected_comm_ids, comm_levels = select_partitions(self.L_allresults, max_nvi=max_nvi,
                                                                    window_size=window_size,basin_radius=basin_radius)
            self.L_allresults['selected_partitions'] = selected_partitions
            self.L_allresults['selected_comm_ids'] = selected_comm_ids
            self.L_allresults['comm_levels'] = comm_levels
            self.L_allresults['onehot_ls'] = get_onehot_ls(self.L_allresults['selected_comm_ids'])

        elif lr == "R":
            selected_partitions, selected_comm_ids, comm_levels = select_partitions(self.R_allresults, max_nvi=max_nvi,
                                                                    window_size=window_size,basin_radius=basin_radius)
            self.R_allresults['selected_partitions'] = selected_partitions
            self.R_allresults['selected_comm_ids'] = selected_comm_ids
            self.R_allresults['comm_levels'] = comm_levels
            self.R_allresults['onehot_ls'] = get_onehot_ls(self.R_allresults['selected_comm_ids'])

    def Draw_multiscale_umap(self,cluster_input="allgenes",spatial=False,save=None):
        """
        Draw umap of hierarchical clustering 

        :param cluster_input is the input user wants to use for drawing umap. It can be "allgenes","L","R", or "userinput"
        :return multiscale umap 
        """        
        # cluster_input:"allgenes"/"L"/"R"/"userinput"
        if cluster_input == "userinput":
            draw_multiscale_umap(cluster_input='allgenes', adata=self.adata, all_results=self.L_allresults, save=save, spatial=spatial)
        if cluster_input == "allgenes":
            draw_multiscale_umap(cluster_input='allgenes', adata=self.adata, all_results=self.L_allresults, save=save, spatial=spatial)
        elif cluster_input == "L":
            draw_multiscale_umap(cluster_input='L', adata=self.adata, all_results=self.L_allresults, save=save, spatial=spatial)
        elif cluster_input == "R":
            draw_multiscale_umap(cluster_input='R', adata=self.adata, all_results=self.R_allresults, save=save, spatial=spatial)

    def Draw_annotations_umap(self):
        """
        Draw umap of cell type annotations

        :return multiscale umap 
        """                
        sc.pl.umap(self.adata, color="annotations", save=False)

    def Detect_specific_LRs(self, topN=20):
        """
        Detect specific ligands and receptors

        :param is the desired number of specific ligand-receptor pairs
        :return list of specific ligand-receptor pairs
        """        
        L_wilcox_dict = dict()
        R_wilcox_dict = dict()
        for gene in self.L_ls:
            gene_exp = compute_genes_geo_avg_vector(self.ligand_exp_dict[gene])
            wilcox_score = get_wilcox_score(gene_exp, self.L_allresults['onehot_ls'])
            L_wilcox_dict[gene] = wilcox_score
        for gene in self.R_ls:
            gene_exp = compute_genes_geo_avg_vector(self.receptor_exp_dict[gene])
            wilcox_score = get_wilcox_score(gene_exp, self.R_allresults['onehot_ls'])
            R_wilcox_dict[gene] = wilcox_score

        columns = ['L_pval', 'L_scale', 'L_id']
        all_LR_filtered = self.all_LR_filtered
        for i, column in enumerate(columns):
            all_LR_filtered[column] = all_LR_filtered['Ligand'].map(lambda x: L_wilcox_dict[x][i])
        columns = ['R_pval', 'R_scale', 'R_id']
        for i, column in enumerate(columns):
            all_LR_filtered[column] = all_LR_filtered['Receptor'].map(lambda x: R_wilcox_dict[x][i])

        all_LR_filtered['specificity'] = all_LR_filtered.L_pval * all_LR_filtered.R_pval
        all_LR_filtered_sorted = self.all_LR_filtered.sort_values('specificity')

        return all_LR_filtered_sorted[0:topN]

    def Draw_CCC_LR(self,ligand,receptor,CCC_threshold=0.4):
        """
        Draw_CCC_LR draws the CCC between a pair of ligand and receptor between ligand clustering and receptor clustering

        :param ligand/receptor is a list of ligands/receptors
        :param CCC_threshold is the threshold of CCC, interactions with strength below it are filtered
        :return: visualization of CCC 
        """
        L_selected_comm_ids = self.L_allresults['selected_comm_ids']
        CCC_mtx = get_CCC_mtx(self.all_LR_filtered, ligand, receptor, self.L_allresults['onehot_ls'], self.R_allresults['onehot_ls'], self.ligand_exp_dict,
                                 self.receptor_exp_dict, self.cofactor_exp_dict, CCC_threshold)

        CCC_mtx = np.concatenate(
            [np.concatenate([CCC_mtx[(l_level, r_level)] for l_level in range(len(self.L_allresults['onehot_ls']))], axis=0) for r_level
             in range(len(self.R_allresults['onehot_ls']))], axis=1)
        draw_CCC(self.L_allresults['onehot_ls'], self.R_allresults['onehot_ls'], CCC_mtx, L_node_mapping_dict=None, R_node_mapping_dict=None,
                    L_is_general=True, R_is_general=True, CCC_linewidth=3, CCC_nodesize=1)

    def Cluster_LRs(self,LR_ls):
        """
        Cluster specific ligand-receptor pairs

        :param LR_ls is the list of ligand-receptor pairs to be clustered         
        :return clustering of ligand-receptor pairs
        """        
        L_exp_in_clusters, R_exp_in_clusters = get_lr_exp_in_clusters(LR_ls, self.L_allresults['onehot_ls'],self.R_allresults['onehot_ls'], self.ligand_exp_dict,
                                                                      self.receptor_exp_dict)
        u, kmeans = cluster_exp_ls(L_exp_in_clusters, R_exp_in_clusters, self.L_allresults, self.R_allresults,
                        self.L_allresults['comm_levels'],self.R_allresults['comm_levels'])

        u[:, 0] = (u[:, 0] - np.min(u[:, 0])) / (np.max(u[:, 0]) - np.min(u[:, 0]))
        u[:, 1] = (u[:, 1] - np.min(u[:, 1])) / (np.max(u[:, 1]) - np.min(u[:, 1]))

        fig = plt.figure()
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        scatter = ax.scatter(u[:, 0], u[:, 1], c=kmeans.labels_)

        sampled_id = np.random.choice(len(LR_ls), size=len(LR_ls), replace=False)
        y_error = np.random.normal(0, 2e-3, len(LR_ls))
        for i, LR_pair in enumerate(LR_ls):
            if i in sampled_id:
                ax.annotate(LR_pair, (u[i, 0], u[i, 1]), textcoords='offset fontsize', fontsize=7)
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc='center left', bbox_to_anchor=(1, 0.5), title="LR groups", alignment='left')
        ax.add_artist(legend1)

    def Detect_active_pathways(self):
        """
        Detect active CCC pathways

        :return active pathways involved in CCC
        """        
        pathways = np.unique(self.all_LR_filtered['pathway_name'])
        pathway_total_CCC_ls = []
        for pathway in pathways:
            pathway_ligands, pathway_receptors = get_pathway_genes(self.all_LR_filtered, pathway)
            CCC_mtx = get_CCC_mtx(self.all_LR_filtered, pathway_ligands, pathway_receptors, self.L_allresults['onehot_ls'],
                                  self.R_allresults['onehot_ls'],self.ligand_exp_dict, self.receptor_exp_dict,
                                     self.cofactor_exp_dict, CCC_threshold=0.3)
            CCC_mtx = np.concatenate(
                [np.concatenate([CCC_mtx[(l_level, r_level)] for l_level in range(len(self.L_allresults['onehot_ls']))], axis=0) for
                 r_level in range(len(self.R_allresults['onehot_ls']))], axis=1)
            pathway_total_CCC_ls.append(np.sum(CCC_mtx))
        self.active_pathways = pathways[np.where(np.asarray(pathway_total_CCC_ls) > 0)[0]]

        return self.active_pathways

    def Draw_CCC(self,pathway,CCC_threshold=0.4,save=None):
        """
        Draw CCC between hierarchical clusters

        :param pathway is the pathway to visualize
        :param CCC_threshold is the threshold of CCC strengths to be visualized
        :return visualization of hierarchical CCC 
        """        
        # save = will save to './figures/{save}'
        L_onehot_ls = self.L_allresults['onehot_ls']
        R_onehot_ls = self.R_allresults['onehot_ls']
        pathway_ligands, pathway_receptors = get_pathway_genes(self.all_LR_filtered, pathway)
        CCC_mtx = get_CCC_mtx(self.all_LR_filtered, pathway_ligands, pathway_receptors, L_onehot_ls, R_onehot_ls,
                                 self.ligand_exp_dict, self.receptor_exp_dict,
                                 self.cofactor_exp_dict, CCC_threshold=CCC_threshold)
        CCC_mtx = np.concatenate(
            [np.concatenate([CCC_mtx[(l_level, r_level)] for l_level in range(len(L_onehot_ls))], axis=0) for r_level in
             range(len(R_onehot_ls))], axis=1)
        draw_CCC(L_onehot_ls, R_onehot_ls, CCC_mtx, L_node_mapping_dict=None, R_node_mapping_dict=None,
                    L_is_general=True, R_is_general=True, CCC_linewidth=3, CCC_nodesize=1, save=save)

    def Cluster_pathways(self,nclusters=3):
        """
        Cluster active CCC pathways

        :param nclusters is the desired number of clusters of pathways
        :return visualiztion of active CCC pathways after clustering
        """        
        L_onehot_ls = self.L_allresults['onehot_ls']
        R_onehot_ls = self.R_allresults['onehot_ls']
        L_Markov_time_ls = get_Markov_time_ls(self.L_allresults, self.L_allresults['comm_levels'])
        R_Markov_time_ls = get_Markov_time_ls(self.R_allresults, self.R_allresults['comm_levels'])
        pathway_L_exp_ls = []
        pathway_R_exp_ls = []
        for pathway in self.active_pathways:
            pathway_ligands, pathway_receptors = get_pathway_genes(self.all_LR_filtered, pathway)
            pathway_ligands_gene_exp = np.zeros(self.ncells)
            pathway_receptors_gene_exp = np.zeros(self.ncells)
            for pathway_ligand in pathway_ligands:
                pathway_ligands_gene_exp += compute_genes_geo_avg_vector(self.ligand_exp_dict[pathway_ligand])
            for pathway_receptor in pathway_receptors:
                pathway_receptors_gene_exp += compute_genes_geo_avg_vector(self.receptor_exp_dict[pathway_receptor])
            pathway_L_exp_ls.append(pathway_ligands_gene_exp)
            pathway_R_exp_ls.append(pathway_receptors_gene_exp)
        pathway_L_exp_in_clusters = []  # ligand expression average in each multiscale cluster for all pathways
        pathway_R_exp_in_clusters = []
        for k in range(len(self.active_pathways)):
            pathway_L_k = []
            pathway_R_k = []
            for i in range(len(L_onehot_ls)):
                pathway_L_k.append(np.matmul(L_onehot_ls[i], pathway_L_exp_ls[k]) / np.sum(L_onehot_ls[i], axis=1))
            for i in range(len(R_onehot_ls)):
                pathway_R_k.append(np.matmul(R_onehot_ls[i], pathway_R_exp_ls[k]) / np.sum(R_onehot_ls[i], axis=1))
            pathway_L_exp_in_clusters.append(np.concatenate(pathway_L_k, axis=0))
            pathway_R_exp_in_clusters.append(np.concatenate(pathway_R_k, axis=0))

        pathway_LR_exp_in_clusters = [np.concatenate((pathway_L, pathway_R), axis=0) for pathway_L, pathway_R in
                                      zip(pathway_L_exp_in_clusters, pathway_R_exp_in_clusters)]
        u, kmeans = cluster_exp_ls(pathway_L_exp_in_clusters, pathway_R_exp_in_clusters, self.L_allresults,
                                   self.R_allresults,self.L_allresults['comm_levels'],self.R_allresults['comm_levels'],
                                    nclusters=nclusters)
        fig, ax = plt.subplots()
        ax.scatter(u[:, 0], u[:, 1], c=kmeans.labels_)
        sampled_id = np.random.choice(len(self.active_pathways), size=int(len(self.active_pathways)), replace=False)
        for i, pathway in enumerate(self.active_pathways):
            if i in sampled_id:
                ax.annotate(pathway, (u[i, 0], u[i, 1]), fontsize=6)

        # plt.savefig('/Users/xinyiwang/Desktop/pathway_clustering.pdf')

    def jaccard_dist(self,comm_ids,celltype_annotations,save=False):
        return jaccard_dist(comm_ids, celltype_annotations, save=save)

