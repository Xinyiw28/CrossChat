import pandas as pd
import numpy as np
import scanpy as sc
import scipy
import scipy.stats
from umap.umap_ import smooth_knn_dist,nearest_neighbors,compute_membership_strengths
from pygenstability import run
from pygenstability.optimal_scales import identify_optimal_scales
from sklearn.metrics.cluster import adjusted_rand_score
import os
import commot as ct
from sklearn.cluster import KMeans
import umap

def load_files(species='human'):
    """
    load_files loads the files needed for CCC calculation

    :return: all_LR is the set of LR pairs in the database
        cofactor_input is the set of cofactors of LR pairs
        complex_input is the set of genes corresponding to each complex gene unit
        """
    if species == 'mouse':
        all_LR = pd.read_csv('../src/crosschat/Input_files/mouse_CellChatDB_LR_pairs.csv', delimiter=',')
        all_LR_cap = all_LR.copy()
        all_LR_cap[['Ligand', 'Receptor']] = all_LR_cap[['Ligand', 'Receptor']].applymap(lambda s: s.upper())
        all_LR = all_LR_cap.copy()
        cofactor_input = pd.read_csv('../src/crosschat/Input_files/mouse_cofactor_input.csv', delimiter=',')
        cofactor_input_cap = cofactor_input[[f'cofactor{i}' for i in range(1, 17)]].applymap(lambda s: str(s).upper())
        cofactor_input_cap['name'] = cofactor_input['name']
        cofactor_input = cofactor_input_cap.copy()
        complex_input = pd.read_csv('../src/crosschat/Input_files/mouse_complex_input.csv',delimiter=',').astype(str)
        complex_input_cap = complex_input[[f'subunit_{i}' for i in range(1, 5)]].applymap(lambda s: str(s).upper())
        complex_input_cap['name'] = complex_input['name'].map(lambda s: str(s).upper())
        complex_input = complex_input_cap.copy()
    if species == 'human':
        all_LR = pd.read_csv('../src/crosschat/Input_files/human_CellChatDB_LR_pairs.csv', delimiter=',')
        all_LR_cap = all_LR.copy()
        all_LR_cap[['Ligand', 'Receptor']] = all_LR_cap[['Ligand', 'Receptor']].applymap(lambda s: s.upper())
        all_LR = all_LR_cap.copy()
        cofactor_input = pd.read_csv('../src/crosschat/Input_files/human_cofactor_input.csv', delimiter=',')
        cofactor_input_cap = cofactor_input[[f'cofactor{i}' for i in range(1, 17)]].applymap(lambda s: str(s).upper())
        cofactor_input_cap['name'] = cofactor_input['name']
        cofactor_input = cofactor_input_cap.copy()
        complex_input = pd.read_csv('../src/crosschat/Input_files/human_complex_input.csv',delimiter=',').astype(str)
        complex_input_cap = complex_input[[f'subunit_{i}' for i in range(1, 5)]].applymap(lambda s: str(s).upper())
        complex_input_cap['name'] = complex_input['name'].map(lambda s: str(s).upper())
        complex_input = complex_input_cap.copy()
    return all_LR, cofactor_input, complex_input

def prepare_gene_exp_dict(L_ls,R_ls,separate_L,separate_R,cofactor_input,ds_geneNames,ds_mtx):
    """
    prepare_gene_exp_dict obtains the dictionary of ligands, receptors,and cofactors

    :return: ligand/receptor/cofactor_exp_dict is the dictionary of ligand/receptor/cofactor expression,
    """
    ligand_exp_dict = {}
    for i, ligand in enumerate(L_ls):
        ligand_exp_dict[ligand.upper()] = []
        for gene in separate_L[i]:
            ligand_exp_dict[ligand.upper()].append(
                get_gene_exp(gene=gene, ds_geneNames=ds_geneNames, ds_mtx=ds_mtx).reshape(1, -1))
        ligand_exp_dict[ligand.upper()] = np.concatenate(ligand_exp_dict[ligand.upper()], axis=0)

    receptor_exp_dict = {}
    for i, receptor in enumerate(R_ls):
        receptor_exp_dict[receptor.upper()] = []
        for gene in separate_R[i]:
            receptor_exp_dict[receptor.upper()].append(
                get_gene_exp(gene=gene, ds_geneNames=ds_geneNames, ds_mtx=ds_mtx).reshape(1, -1))
        receptor_exp_dict[receptor.upper()] = np.concatenate(receptor_exp_dict[receptor.upper()], axis=0)

    cofactor_exp_dict = {}
    for i in range(len(cofactor_input)):
        cofactor = cofactor_input['name'][i]
        cofactor_exp_dict[cofactor] = []
        genes = set(cofactor_input.iloc[i].values) - {'NAN'} - {cofactor}
        for gene in genes:
            cofactor_exp_dict[cofactor].append(
                get_gene_exp(gene=gene, ds_geneNames=ds_geneNames, ds_mtx=ds_mtx).reshape(1, -1))
        cofactor_exp_dict[cofactor] = np.concatenate(cofactor_exp_dict[cofactor], axis=0)

    return ligand_exp_dict,receptor_exp_dict,cofactor_exp_dict

def create_adata(ds_mtx=None, ds_geneNames:pd.DataFrame=None, ds_annotations:pd.DataFrame=None):
    # ds_mtx: ngenes*ncells count matrix
    # ds_geneNames: dataframe of gene names
    # ds_annotations: dataframe of cell type annotations
    # normalized: whether ds_mtx is lognormalized
    # scaled: whether ds_mtx is scaled
    # input: allgenes/L/R
    adata = sc.AnnData(ds_mtx.T)
    if ds_annotations is not None:
        adata.obs['celltype_annotations'] = ds_annotations.values
    if not isinstance(ds_geneNames, list):
        adata.var_names = ds_geneNames.values
    else:
        adata.var_names = ds_geneNames
    return adata

def prepare_adata(adata, normalize=False, scale=False, input='allgenes'):
    #
    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if input=='allgenes':
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    if scale == True:
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata

def obtain_spatial_pca(pca_embedding,spatial_pos,w=0.5):
    #w: weight of spatial importance
    row_norms = np.linalg.norm(pca_embedding, axis=1)
    scaled_pca = pca_embedding / np.mean(row_norms)
    shifted_pos = spatial_pos - np.mean(spatial_pos, axis=0)
    shifted_pos_row_norms = np.linalg.norm(shifted_pos, axis=1)
    scaled_pos = shifted_pos / np.mean(shifted_pos_row_norms)
    spatial_pca = np.concatenate((scaled_pca, w * scaled_pos), axis=1)

    return spatial_pca

def multiscale_clustering(pca_embedding, min_scale=-1, max_scale=4, n_scale=100, n_tries=20, k=15):

    knn_indices, knn_dists, rp_trees = nearest_neighbors(pca_embedding, n_neighbors=15, metric="cosine",
                                                         metric_kwds=None, angular=False, random_state=None)
    ncells = pca_embedding.shape[0]
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, k=k, n_iter=64, local_connectivity=1.0, bandwidth=1.0)
    rows, cols, vals, dists = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos, return_dists=False)
    A = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(ncells, ncells))
    S = 0.0000001 * scipy.sparse.eye(ncells,k=-1)
    A = A + S
    B = A.transpose(axes=None, copy=False)
    A_sym = A.maximum(B)
    all_results = run(A_sym, min_scale=min_scale, max_scale=max_scale, n_scale=n_scale, n_tries=n_tries)

    return all_results

def select_partitions(all_results,max_nvi=0.1,window_size=10,basin_radius=10):
    ncells = all_results['community_id'][0].shape[0]
    all_results = identify_optimal_scales(all_results,max_nvi=max_nvi,window_size=window_size, basin_radius=basin_radius)
    selected_partitions = [int(partition) for partition in all_results['selected_partitions']]
    final_selected_partitions = [partition for partition in selected_partitions if 0 < max(all_results['community_id'][partition]) < ncells - 1]
    selected_partitions = []
    selected_partitions.append(final_selected_partitions[0])
    for i in range(len(final_selected_partitions) - 1):
        P1 = all_results['community_id'][final_selected_partitions[i + 1]]
        P2 = all_results['community_id'][final_selected_partitions[i]]
        ARI = adjusted_rand_score(P1, P2)
        if ARI < 0.95:
            selected_partitions.append(final_selected_partitions[i + 1])

    n_scales = len(selected_partitions)
    comm_levels = [0]*n_scales 

    for i in range(n_scales):
        l = selected_partitions[i] # the timestep of the selected partition
        comm_levels[i] = max(all_results['community_id'][l])+1
    comm_levels = [int(x) for x in comm_levels]

    final_selected_partitions = []
    i = 0
    j = 0
    while i < len(comm_levels):
        if j < len(comm_levels):
            np = comm_levels[i]
            nl = comm_levels[j]
            if np <= nl:
                j += 1
            else:
                final_selected_partitions.append(selected_partitions[i])
                i = j
        else:
            final_selected_partitions.append(selected_partitions[i])
            break

    final_comm_levels = [int(max(all_results['community_id'][final_selected_partitions[i]])+1) for i in range(len(final_selected_partitions))]
    final_comm_levels = final_comm_levels[::-1]
    final_selected_partitions = final_selected_partitions[::-1]
    final_selected_comm_ids = [all_results['community_id'][final_selected_partitions[i]] for i in range(len(final_selected_partitions))]

    return final_selected_partitions, final_selected_comm_ids, final_comm_levels

def save_allresults(all_results,dir):
    PARENT_DIR = os.getcwd()
    save_dir = os.path.join(PARENT_DIR, dir)
    os.makedirs(save_dir, exist_ok=True)

    for key in list(all_results.keys()):
        if key not in ["run_params", "onehot_ls"]:
            np.save(f"{save_dir}/{key}.npy".format(key=key), all_results[key])
        elif key == 'onehot_ls':
            with open(f"{save_dir}/{key}.npz".format(key=key), 'wb') as f:
                pickle.dump(all_results[key], f)
    return None


def load_allresults(dir):
    keys = ['scales',
            'number_of_communities',
            'stability',
            'community_id',
            'NVI',
            'ttprime',
            'block_detection_curve',
            'selected_partitions']
    all_results = dict()
    for key in keys:
        all_results[key] = np.load(f"{dir}/{key}.npy")
    return all_results

def get_cluster_results(comm_ids):
    allresults = {}
    allresults['selected_comm_ids'] = comm_ids
    allresults['comm_levels'] = [int(max(comm_id))+1 for comm_id in comm_ids]
    allresults['onehot_ls'] = get_onehot_ls(comm_ids)

    return allresults

def filter_all_LR(all_LR, ligand_exp_dict, receptor_exp_dict, ncells, threshold = 0.005):

    all_LR_filtered = pd.DataFrame()
    for i in range(len(all_LR)):
        ligand = all_LR['Ligand'][i]
        receptor = all_LR['Receptor'][i]
        ligand_exp = ligand_exp_dict[ligand]
        receptor_exp = receptor_exp_dict[receptor]
        ligand_support = np.ones(ncells)
        receptor_support = np.ones(ncells)
        for j in range(len(ligand_exp)):
            ligand_support *= ligand_exp[j]
        for j in range(len(receptor_exp)):
            receptor_support *= receptor_exp[j]
        ligand_support[ligand_support > 0] = 1
        receptor_support[receptor_support > 0] = 1
        if np.sum(ligand_support) > threshold*ncells and np.sum(receptor_support) > threshold*ncells:
            all_LR_filtered = pd.concat([all_LR_filtered, all_LR.iloc[i].to_frame().T], ignore_index=True)

    return all_LR_filtered

def prepare_lr_sup_mtx(all_LR,complex_input,lr='L',ds_mtx=None,ds_geneNames=None):
    """
    prepare_lr_sup_mtx obtain the support matrix of ligands/receptors

    :param all_LR: LR interactions from CellChatDB
    :param complex_input: complex_input file from CellChatDB
    :param lr: 'L' or 'R'
    :param ds_mtx: gene expression matrix
    :param ds_geneNames: dataframe of gene names, with column name as 0
    :return: lr_sup_mtx is the support matrix of ligands/receptors,
        all_LR_lr_ind is dataframe of list of indices of ligands/receptors in all_LR,
        lr_ls is list of unique ligands/receptors, in which ligand/receptor complex uses its complex name
        separate_lr is list of unique ligands/receptors, in which ligand/receptor complex separates into different gene names
    """

    ncells = ds_mtx.shape[1]

    if lr == 'L':
        lr_long = 'Ligand'
    else:
        lr_long = 'Receptor'

    separate_lr = []
    lr_ls = all_LR[lr_long].drop_duplicates().values
    for gene in lr_ls:
        if gene in complex_input['name'].values:
            ind = np.where(complex_input['name'].values == gene)[0][0]
            separate_lr.append(list(set(complex_input.iloc[ind]) - {'NAN'} - {gene}))
        else:
            separate_lr.append([gene])

    lr_indices = dict()
    for name in separate_lr:
        for gene in name:
            if len(ds_geneNames.loc[ds_geneNames[0] == gene.upper()].index) > 0:
                lr_indices[gene] = ds_geneNames.loc[ds_geneNames[0] == gene.upper()].index[0]
            else:
                lr_indices[gene] = 1e6

    lr_ind = dict()
    for i,gene in enumerate(lr_ls):
        lr_ind[gene] = ([lr_indices[k] for k in separate_lr[i]])

    all_LR_lr_ind = all_LR[lr_long].apply(lambda i: lr_ind[i]) #indices of l/r in ds_geneNames
    lr_sup_mtx = np.ones((len(separate_lr), ncells))

    for i, gene_name in enumerate(separate_lr):
        for gene in gene_name:
            if lr_indices[gene] < 1e6:
                lr_sup_mtx[i, :] *= np.squeeze(np.asarray(ds_mtx[lr_indices[gene]].todense()))
            else:
                lr_sup_mtx[i, :] *= 0
    lr_sup_mtx[lr_sup_mtx > 0] = 1

    return lr_sup_mtx, all_LR_lr_ind, lr_ls, separate_lr

def prepare_lr_union_sup_mtx(all_LR_filtered, L_sup_mtx, R_sup_mtx, L_ls,R_ls):
    """
    prepare_lr_union_sup_mtx obtain the support matrix of union of ligand and receptor in each ligand-receptor pair of all_LR_filtered

    :param all_LR_filtered: filtered LR interactions from CellChatDB, only including LR pairs that both exist in the dataset
    :param L_sup_mtx: support matrix of all lignds
    :param L_ls: list of ligands, in complex name form
    :return: LR_sup_mtx is the support matrix of union of ligand and receptor pair, sorted according to all_LR_filtered
    """

    ncells = L_sup_mtx[0].shape[0]
    LR_sup_mtx = np.ones((len(all_LR_filtered), ncells))

    for i in range(len(all_LR_filtered)):
        ligand = all_LR_filtered['Ligand'].iloc[i]
        receptor = all_LR_filtered['Receptor'].iloc[i]
        ligand_ind = np.where(L_ls==ligand)[0][0]
        receptor_ind = np.where(R_ls==receptor)[0][0]
        lr_sup = L_sup_mtx[ligand_ind] + R_sup_mtx[receptor_ind]
        lr_sup[lr_sup>0] = 1
        LR_sup_mtx[i] = lr_sup

    return LR_sup_mtx

def draw_multiscale_umap(cluster_input,adata,all_results,save=None,spatial=False):
    """
    draw_multiscale_umap draws the umap of multiscale clustering results
    :param cluster_input: 'L','R','allgenes'
    :param adata:
    :param all_results:
    :param comm_levels:
    :param selected_partitions:
    :param save: directory (example: /Users/abc/Desktop/directory_name)
    """
    comm_levels = all_results['comm_levels']
    n_scales = len(comm_levels)
    for i in range(n_scales):
        j = int(comm_levels[i])  # number of clusters at the selected scale
        globals()[f'communities_{j}_labels'] = all_results['selected_comm_ids'][i]
    ds_community = pd.DataFrame()  # Dataframe of community labels
    for i in range(n_scales):
        j = comm_levels[i]
        ds_community[f'community_{j}'] = globals()[f'communities_{j}_labels']
        adata.obs[f'{cluster_input}_community_{j}'] = pd.Series(globals()[f'communities_{j}_labels'], dtype="category").values

    onehot_ls = get_onehot_ls(all_results['selected_comm_ids'])
    node_color_ls = get_general_tree_colors(onehot_ls)

    if spatial == False:
        for i,comm_level in enumerate(comm_levels):
            adata.uns[f'{cluster_input}_community_{comm_level}_colors'] = node_color_ls[i]
            if save == True:
                sc.pl.umap(adata, color=f"{cluster_input}_community_{comm_level}", save=save)
            elif save:
                filename = save + f'_{cluster_input}_community_{comm_level}.pdf'
                sc.pl.umap(adata, color=f"{cluster_input}_community_{comm_level}", save=filename)
            else:
                sc.pl.umap(adata, color=f"{cluster_input}_community_{comm_level}")
    else:
        for i,comm_level in enumerate(comm_levels):
            adata.uns[f'{cluster_input}_community_{comm_level}_colors'] = node_color_ls[i]
            if save ==True:
                sc.pl.embedding(adata, basis="spatial", color=f"{cluster_input}_community_{comm_level}", save=save)
            elif save:
                filename = save + f'_{cluster_input}_community_{comm_level}.pdf'
                sc.pl.embedding(adata, basis="spatial", color=f"{cluster_input}_community_{comm_level}", save=filename)
            else:
                sc.pl.embedding(adata, basis="spatial", color=f"{cluster_input}_community_{comm_level}")

def get_general_tree_colors(onehot_ls):
    comm_levels = [len(onehot_ls[i]) for i in range(len(onehot_ls))]
    rel_i_j = get_rel_i_j_ls(onehot_ls)
    default_color_ls = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#bcbd22', '#17becf',
                        '#8A181A','#F04E4B','#FDE3E0','#8ACC99','#97AFDB','#997E88','#F8A993','#2DBA9D']

    node_color_ls = []

    for i in range(len(comm_levels)):
        if i == 0:
            node_color_ls.append([default_color_ls[j % 18] for j in range(comm_levels[-1])])
        else:
            node_color_ls.append([node_color_ls[-1][np.where(rel_i_j[0][-1 * i] == j)[0][0]] for j in range(comm_levels[-1 - i])])

    return node_color_ls[::-1]


def get_gene_exp(gene,ds_geneNames,ds_mtx):
    """
    get_gene_exp obtains gene expression of input gene from gene expression matrix.

    :param gene: the input gene from which to obtain its gene expression
    :param ds_geneNames: dataframe of gene names of gene expression matrix
    :param ds_mtx: gene expression matrix
    :return: gene expression of input gene
    """
    ncells = np.shape(ds_mtx)[1]
    gene_ind = np.where(ds_geneNames[0]==gene.upper())[0]

    if len(gene_ind) == 0:
        gene_exp = np.zeros(ncells)
    else:
        gene_exp = np.asarray(ds_mtx[gene_ind].todense())

    return gene_exp

def get_onehot_ls(selected_comm_ids):
    """
    get_onehot_ls Obtain onehot list of each cluster based on a list of selected partitions.

    :param selected_partitions: list of arrays of selected partitions, each array represents a partition at one scale
    :return: list of onehot vectors of selected partitions
    """

    ncells = selected_comm_ids[0].shape[0]
    onehot_ls = []
    for partition in selected_comm_ids:
        n_clusters = int(max(partition)+1)
        onehot_of_partition = []
        for i in range(n_clusters):
            onehot = np.zeros((1,ncells))
            onehot[0][np.where(partition==i)[0]] = 1
            onehot_of_partition.append(onehot)
        onehot_ls.append(np.concatenate(onehot_of_partition,axis=0))

    return onehot_ls

def get_wilcox_score(gene_exp,onehot_ls):
    """
    get_wilcox_score runs Wilcoxan rank sum test on the input gene of each multiscale cluster and returns its
    minimum score, the scale and cluster id of the multiscale cluster that the gene is most specific to

    :param gene_exp: gene expression
    :param onehot_ls: list of onehot vectors of selected partitions
    :return: wilcox_score is the triple [min_score,argmin_scale,argmin_cluster]
    """
    w_score_ls = []
    for i in range(len(onehot_ls)):
        for j in range(len(onehot_ls[i])):
            onehot1 = gene_exp[np.where(onehot_ls[i][j]==1)[0]]
            onehot2 = gene_exp[np.where(onehot_ls[i][j]==0)[0]]
            w_score = scipy.stats.ranksums(onehot1, onehot2, alternative='greater')[1]
            w_score_ls.append(w_score)
    min_score = np.min(w_score_ls)
    argmin = np.argmin(w_score_ls)
    # Check the scale and cluster id where the min_score occurs at
    scale = 0
    while argmin >= len(onehot_ls[scale]) and scale < len(onehot_ls):
        argmin -= len(onehot_ls[scale])
        scale += 1
    id = argmin

    argmin_scale = scale
    argmin_cluster = id
    wilcox_score = [min_score, argmin_scale, argmin_cluster]
    return wilcox_score

def assign_top_marker_to_cluster(adata,cluster_input,comm_levels):
    """
    assign_top_marker_to_cluster obtains the list of top marker to each multiscale cluster

    :param adata:
    :param cluster_input: 'L'/'R'/'allgenes'
    :param comm_levels:
    :return: top_marker_ls
    """
    top_marker_ls = []
    for comm_level in comm_levels:
        sc.tl.rank_genes_groups(adata, f'{cluster_input}_community_{comm_level}', method='wilcoxon')
        top_marker_ls.append(list(adata.uns['rank_genes_groups']['names'])[0])
    return top_marker_ls

def find_lr_intersected_clusters(lr_inds, lr_onehot_ls, lr_sup_mtx):
    """
    find_lr_intersected_clusters obtain ligands/receptors intersected clusters.

    :param lr: 'L' or 'R'
    :param lr_inds: list of indices of ligand or receptor in L_ls or R_ls
    :param lr_onehot_ls: list of onehot vectors of l/r multiscale clusters
    :param lr_sup_mtx: L_sup_mtx or R_sup_mtx
    :return: list of arrays of onehot vectors of ligands/receptors intersected clusters of lr_onehot_ls
    """

    def find_intersection(lr_inds,scale):
        ncells = lr_sup_mtx[0].shape[0]
        lr_support = np.zeros(lr_sup_mtx[0].shape[0])

        for lr_ind in lr_inds:
            lr_support += lr_sup_mtx[lr_ind]

        lr_support[lr_support>0] = 1

        scale_onehot = lr_onehot_ls[scale]
        lr_intersected_scale_onehot = []
        lr_original_scale_onehot = []

        for i in range(len(scale_onehot)):

            lr_in_cluster_prop = np.sum(scale_onehot[i] * lr_support)/np.sum(scale_onehot[i]) #proportion of cells that contain lrs among all cells in the cluster
            lr_cluster_in_lrs_prop = np.sum(scale_onehot[i] * lr_support)/np.sum(lr_support)  #proportion of cells that contain lrs that is in cluster among all cells containing lrs

            if lr_in_cluster_prop < 0.05 or lr_cluster_in_lrs_prop < 0.05:
                lr_intersected_scale_onehot.append(scale_onehot[i])
                lr_original_scale_onehot.append([scale_onehot[i],0])
            else:
                l1 = scale_onehot[i] * lr_support
                l2 = scale_onehot[i] * (1-lr_support)
                if np.sum(l1)>ncells*0.005 and np.sum(l2)>ncells*0.005:
                    lr_intersected_scale_onehot.append(l1)
                    lr_original_scale_onehot.append([scale_onehot[i], 1])
                    lr_intersected_scale_onehot.append(l2)
                    lr_original_scale_onehot.append([scale_onehot[i], 0])
                else:
                    lr_intersected_scale_onehot.append(scale_onehot[i])
                    lr_original_scale_onehot.append([scale_onehot[i], 0])

        return np.asarray(lr_intersected_scale_onehot), lr_original_scale_onehot

    # Determine if the lr is contained in one cluster in input scale
    def is_included_in_scale(lr_inds, scale):

        is_contained = False

        lr_support = np.zeros_like(lr_sup_mtx[0])

        for lr_ind in lr_inds:
            lr_support += lr_sup_mtx[lr_ind]

        lr_support[lr_support>0] = 1

        scale_onehot = lr_onehot_ls[scale]

        for i in range(len(scale_onehot)):
            lr_cluster_in_lrs_prop = np.sum(scale_onehot[i] * lr_support)/np.sum(lr_support)
            if lr_cluster_in_lrs_prop > 0.95:
                is_contained = True

        return is_contained

    lr_intersected_onehot_ls = []
    lr_original_onehot_ls = []

    n_scales = len(lr_onehot_ls)
    scale = 0

    while scale < n_scales:
        while scale < n_scales - 1 and is_included_in_scale(lr_inds, scale + 1):
            lr_intersected_onehot_ls.append(lr_onehot_ls[scale])
            lr_original_onehot_ls.append([[onehot,0] for onehot in lr_onehot_ls[scale]])
            scale += 1
        lr_intersected_scale_onehot, lr_original_scale_onehot = find_intersection(lr_inds, scale)
        lr_intersected_onehot_ls.append(lr_intersected_scale_onehot)
        lr_original_onehot_ls.append(lr_original_scale_onehot)
        scale += 1

    return lr_intersected_onehot_ls, lr_original_onehot_ls


def get_rel_i_j_ls(onehot_ls):
    """
    get_rel_i_j_ls obtain the list of arrays of relationship of multiscale clusters, each array represents
    the correspondence relationship of a finer cluster to a coarser cluster

    :param onehot_ls: the list of onehot vectors of multiscale ligand clusters
    :return: rel_i_j_ls is list of arrays of onehot vectors of relationship of multiscale clusters
    """
    rel_i_j_ls = []
    comm_levels = []
    for i in range(len(onehot_ls)):
        comm_levels.append(onehot_ls[i].shape[0])

    rel_i_j = []  # relationship vectors

    for l, k in enumerate(comm_levels):
        if l < len(comm_levels) - 1:
            i = comm_levels[l]
            j = comm_levels[l + 1]
            rel_j_i = np.zeros((j, i))  # j * i matrix

            for m in range(j):
                for n in range(i):
                    rel_j_i[m, n] = np.dot(onehot_ls[l + 1][m], onehot_ls[l][n])

            rel_j_i_result = rel_j_i.argmax(axis=1)
            rel_i_j.append(rel_j_i_result)

    rel_i_j_ls.append(rel_i_j)

    return rel_i_j_ls

def convert_complex_to_genes_ls(complex_input_cap,gene_complex):
    """
    convert_complex_to_genes_ls obtain the list of separate genes from gene complex;
    if the input is a single gene, then output the list with single gene

    :gene_complex: the gene complex to be separated
    :return: genes is the list of genes corresponding to the gene_complex
    """

    if gene_complex.upper() in complex_input_cap['name'].values:
        ind = np.where(complex_input_cap['name'] == gene_complex.upper())
        genes = list(set(np.asarray(complex_input_cap.iloc[ind].values[0],dtype=str)) - {'NAN'} - {gene_complex})
    else:
        genes = gene_complex

    return genes

def get_pathway_genes(all_LR,pathway):
    """
    get_pathway_genes obtain the list of ligands and receptors in ds_geneNames of the input pathway

    :all_LR: dataframe of ligand-receptor interactions from CellChatDB
    :param pathway: name of pathway
    :return: pathway_ligands is the list of ligands of input pathway, in complex form
    """
    pathway_ind = np.where(all_LR['pathway_name'].values == pathway)[0]
    pathway_ligands = list(set(all_LR['Ligand'].values[pathway_ind]))
    pathway_receptors = list(set(all_LR['Receptor'].values[pathway_ind]))

    return pathway_ligands, pathway_receptors

def get_gene_ind_in_gene_ls(genes,gene_ls):
    """
    get_pathway_genes obtain the list of indices of genes in gene_ls

    :param genes: the gene names, needs to be all upper cased
    :param gene_ls: the list of genes
    :return: gene_ind the list of indices of genes in gene_ls
    """

    gene_ind = []
    for i in range(len(genes)):
        if genes[i].upper() in [x.upper() for x in gene_ls]:
            ind = [x.upper() for x in gene_ls].index(genes[i])
            gene_ind.append(ind)

    return gene_ind

def compute_genes_geo_avg_vector(genes_exp_ls):
    """
    compute_genes_geo_avg_vector obtain the gene expression vector of geometric mean of a list of genes

    :param genes_exp_ls: the list of gene expression of input single genes
    :return: gene_exp is the gene expression vector of geometric mean of the list of input gene expressions
    """
    gene_exp = np.ones(len(genes_exp_ls[0]))

    for i in range(len(genes_exp_ls)):
        gene_exp *= genes_exp_ls[i]

    gene_exp = np.power(gene_exp, 1/len(genes_exp_ls))

    return gene_exp

def compute_ligand_geo_avg(ligand_exp_ls,cluster_onehot):

    L_i_ls = []

    for i in range(len(ligand_exp_ls)):
        L_i_ls.append(np.dot(ligand_exp_ls[i],cluster_onehot)/np.sum(cluster_onehot))

    avg_ligand_exp = np.power(np.prod(L_i_ls), 1/len(L_i_ls))

    return avg_ligand_exp

def compute_cofactor_avg(cofactor_exp_ls,cluster_onehot):

    cofactor_i_ls = []

    if len(cofactor_exp_ls)>0:
        for i in range(len(cofactor_exp_ls)):
            cofactor_i_ls.append(np.dot(cofactor_exp_ls[i],cluster_onehot)/np.sum(cluster_onehot))
        avg_cofactor_exp = np.mean(np.asarray(cofactor_i_ls))
    else:
        avg_cofactor_exp = 0

    return avg_cofactor_exp

def compute_interaction(LR_ind,cluster1_onehot,cluster2_onehot,all_LR,ligand_exp_dict,receptor_exp_dict,cofactor_exp_dict):
#LR_ind: indices of LR pair in all_LR

    ncells = np.shape(cluster1_onehot)[0]
    K = 0.5

    l = all_LR['Ligand'].iloc[LR_ind]
    r = all_LR['Receptor'].iloc[LR_ind]
    ag = all_LR['agonist'].iloc[LR_ind]
    an = all_LR['antagonist'].iloc[LR_ind]
    ra = all_LR['co_A_receptor'].iloc[LR_ind]
    ri = all_LR['co_I_receptor'].iloc[LR_ind]

    ligand_exp_ls = ligand_exp_dict[l]
    receptor_exp_ls = receptor_exp_dict[r]

    if str(ag) == 'nan':
        agonist_exp_ls = [np.zeros(ncells)]
    else:
        agonist_exp_ls = cofactor_exp_dict[ag]
    if str(an) == 'nan':
        antagonist_exp_ls = [np.zeros(ncells)]
    else:
        antagonist_exp_ls = cofactor_exp_dict[an]
    if str(ra) == 'nan':
        receptor_a_exp_ls = [np.zeros(ncells)]
    else:
        receptor_a_exp_ls = cofactor_exp_dict[ra]
    if str(ri) == 'nan':
        receptor_i_exp_ls = [np.zeros(ncells)]
    else:
        receptor_i_exp_ls = cofactor_exp_dict[ri]

    AG_ligand = compute_cofactor_avg(agonist_exp_ls,cluster1_onehot)
    AG_receptor = compute_cofactor_avg(agonist_exp_ls,cluster2_onehot)
    AN_ligand = compute_cofactor_avg(antagonist_exp_ls,cluster1_onehot)
    AN_receptor = compute_cofactor_avg(antagonist_exp_ls,cluster2_onehot)
    RA = compute_cofactor_avg(receptor_a_exp_ls,cluster2_onehot)
    RI = compute_cofactor_avg(receptor_i_exp_ls,cluster2_onehot)
    L = compute_ligand_geo_avg(ligand_exp_ls,cluster1_onehot)
    R = compute_ligand_geo_avg(receptor_exp_ls,cluster2_onehot)*(1+RA)/(1+RI)

    P = (L*R)/(K+L*R)*(1+AG_ligand/(K+AG_ligand))*(1+AG_receptor/(K+AG_receptor))*(K/(K+AN_ligand))*(K/(K+AN_receptor))

    return P

def get_CCC_mtx(all_LR, ligands_ls, receptors_ls, ligand_clusters, receptor_clusters,ligand_exp_dict,receptor_exp_dict,
                cofactor_exp_dict, CCC_threshold):
    """
    get_CCC_mtx obtains total interaction matrix between set of ligands with ligands_indices (in L_ls, can be ligand complex) and receptors
    with receptors_indices (in R_ls), between the input multiscale ligand_clusters and multiscale receptor_clusters

    :ligands_ls: list of ligands
    :ligand_clusters: list of arrays of multiscale ligand clusters
    :return: CCC_mtx is the total interaction matrix
    """

    LR_indices_in_all_LR = [] # List of indices of interacting LR paris in all_LR
    for i in range(len(all_LR)):
        if all_LR['Ligand'].iloc[i] in ligands_ls and all_LR['Receptor'].iloc[i] in receptors_ls:
            LR_indices_in_all_LR.append(i)

    CCC_mtx_ls = []

    for n_LR in LR_indices_in_all_LR:
        LR_CCC = dict()
        for L_scale in range(len(ligand_clusters)):
            for R_scale in range(len(receptor_clusters)):
                n_L_clusters = len(ligand_clusters[L_scale])
                n_R_clusters = len(receptor_clusters[R_scale])
                CCC_mtx = np.zeros((n_L_clusters, n_R_clusters))

                for L_scale_cluster in range(n_L_clusters):
                    for R_scale_cluster in range(n_R_clusters):
                        ligand_onehot = ligand_clusters[L_scale][L_scale_cluster]
                        receptor_onehot = receptor_clusters[R_scale][R_scale_cluster]
                        CCC_mtx[L_scale_cluster, R_scale_cluster] = compute_interaction(n_LR, ligand_onehot,
                                receptor_onehot, all_LR, ligand_exp_dict, receptor_exp_dict, cofactor_exp_dict)
                LR_CCC[(L_scale, R_scale)] = CCC_mtx

        CCC_mtx_ls.append(LR_CCC)

    total_CCC_mtx = dict()
    for L_scale in range(len(ligand_clusters)):
        for R_scale in range(len(receptor_clusters)):
            n_L_clusters = len(ligand_clusters[L_scale])
            n_R_clusters = len(receptor_clusters[R_scale])
            total_CCC_mtx[(L_scale,R_scale)] = np.zeros((n_L_clusters,n_R_clusters))
    for i in range(len(CCC_mtx_ls)):
        for L_scale in range(len(ligand_clusters)):
            for R_scale in range(len(receptor_clusters)):
                total_CCC_mtx[(L_scale,R_scale)] += CCC_mtx_ls[i][(L_scale,R_scale)]
                total_CCC_mtx[(L_scale, R_scale)][total_CCC_mtx[(L_scale, R_scale)] < CCC_threshold] = 0

    return total_CCC_mtx

def get_CCC_mtx_from_COMMOT(adata, l_r_pathway, ligand_clusters, receptor_clusters, CCC_threshold=0, normalize=True):
    #l_r_pathway: a tuple as ['Igf2', 'Igf2r', 'Igf_pathway']
    l_r_pathway_ls = np.array([l_r_pathway],dtype=str)
    df_ligrec = pd.DataFrame(data=l_r_pathway_ls)
    ct.tl.spatial_communication(adata, database_name='user_database', df_ligrec=df_ligrec, dis_thr=200, heteromeric=True)
    commot_mtx_sc = adata.obsp[f'commot-user_database-{l_r_pathway[0]}-{l_r_pathway[1]}'].todense()

    LR_CCC = dict()
    for L_scale in range(len(ligand_clusters)):
        for R_scale in range(len(receptor_clusters)):
            n_L_clusters = len(ligand_clusters[L_scale])
            n_R_clusters = len(receptor_clusters[R_scale])
            CCC_mtx = np.zeros((n_L_clusters, n_R_clusters))

            for L_scale_cluster in range(n_L_clusters):
                for R_scale_cluster in range(n_R_clusters):
                    ligand_onehot = ligand_clusters[L_scale][L_scale_cluster]
                    receptor_onehot = receptor_clusters[R_scale][R_scale_cluster]
                    if normalize:
                        CCC_mtx[L_scale_cluster, R_scale_cluster] = np.sum(
                            commot_mtx_sc[np.nonzero(ligand_onehot)[0], :][:, np.nonzero(receptor_onehot)[0]]) / (np.sum(ligand_onehot) * np.sum(receptor_onehot))
                    else:
                        CCC_mtx[L_scale_cluster, R_scale_cluster] = np.sum(commot_mtx_sc[np.nonzero(ligand_onehot)[0], :][:, np.nonzero(receptor_onehot)[0]])
            CCC_mtx[CCC_mtx < 0] = 0
            LR_CCC[(L_scale, R_scale)] = CCC_mtx

    return LR_CCC

def get_cluster_size_ls(onehot_ls):
    """
    get_lr_size_ls obtains the list of arrays of node sizes of ligand/receptor clusters

    :lr_onehot_ls: L_onehot_ls is the list of arrays of onehot vectors of ligand clusters
    :return: list of arrays of sizes of ligand/receptor clusters
    """
    lr_size_ls = []
    for i in range(len(onehot_ls)):
        size_ls = []
        for j in range(len(onehot_ls[i])):
            size_ls.append(np.sum(onehot_ls[i][j]) / len(onehot_ls[0][0]))
        lr_size_ls.append(size_ls)

    return lr_size_ls

from collections import Counter
def get_lr_frequency_in_trees(orig_lr_indices_in_trees,lr_ls):
    """
    get_lr_frequency_in_trees obtains the dictionary of ligand/receptor frequency in all multiscale ligand/receptor trees

    :orig_lr_indices: the list of original lr indices in all ligand/receptor trees
    :lr_ls: L_ls or R_ls, list of distinct ligands/receptors
    :return: dictionary of ligand/receptor frequency in all multiscale ligand/receptor trees
    """

    count = Counter(orig_lr_indices_in_trees)
    X = [[lr_ls[item],frequency] for item, frequency in count.most_common()]
    lr_freq_dict = dict()
    for x,y in X:
        lr_freq_dict[x] = y

    return lr_freq_dict


def get_Markov_time_ls(all_results,comm_levels):
    """
    get_Markov_time_ls obtain the list of markov times corresponding to selected community levels

    :param all_results: output of multiscale clustering results
    :param comm_levels: selected community levels
    :return: Markov_time_ls is the list of markov times corresponding to selected community levels,
        starting from finest scale preset at markov time 0, ending at coarsest scale (1 cluster)
    """

    Markov_time_ls = []
    '''
    if np.min(all_results['number_of_communities']) == 1:
        step = np.where(all_results['number_of_communities'] == 1)[0][0]
        T = np.log10(all_results['scales'][step])
        Markov_time_ls.append(T)
    else:
        T = np.log10(all_results['scales'][-1])+1
        Markov_time_ls.append(T)
    '''
    for i in comm_levels:
        step = np.where(np.asarray(all_results['number_of_communities']) == i)[0][0]
        T = all_results['scales'][step]
        Markov_time_ls.append(T)

    return Markov_time_ls

def get_lr_exp_in_clusters(LR_ls,L_onehot_ls,R_onehot_ls,ligand_exp_dict,receptor_exp_dict):
    """
    get_lr_exp_in_clusters obtain the list of markov times corresponding to selected community levels

    :param LR_ls: list of [[L1,R1],[L2,R2],...]
    :param all_results: output of multiscale clustering results
    :param comm_levels: selected community levels
    :return: Markov_time_ls is the list of markov times corresponding to selected community levels,
        starting from finest scale preset at markov time 0, ending at coarsest scale (1 cluster)
    """
    L_exp_ls = []
    R_exp_ls = []
    for i in range(len(LR_ls)):
        ligand,receptor = LR_ls[i]
        ligand_exp = compute_genes_geo_avg_vector(ligand_exp_dict[ligand])
        receptor_exp = compute_genes_geo_avg_vector(receptor_exp_dict[receptor])
        L_exp_ls.append(ligand_exp)
        R_exp_ls.append(receptor_exp)

    L_exp_in_clusters = []  # ligand expression average in each multiscale cluster for all pathways
    R_exp_in_clusters = []

    for k in range(len(LR_ls)):
        L_k = []
        R_k = []
        for i in range(len(L_onehot_ls)):
            L_k.append(np.matmul(L_onehot_ls[i], L_exp_ls[k]) / np.sum(L_onehot_ls[i], axis=1))
        for i in range(len(R_onehot_ls)):
            R_k.append(np.matmul(R_onehot_ls[i], R_exp_ls[k]) / np.sum(R_onehot_ls[i], axis=1))
        L_exp_in_clusters.append(np.concatenate(L_k, axis=0))
        R_exp_in_clusters.append(np.concatenate(R_k, axis=0))

    return L_exp_in_clusters,R_exp_in_clusters

def cluster_exp_ls(L_exp_in_clusters,R_exp_in_clusters,L_all_results,R_all_results,L_comm_levels,R_comm_levels,nclusters=3):
    """
    cluster_exp_ls obtain the list of markov times corresponding to selected community levels

    :param all_results: output of multiscale clustering results
    :param comm_levels: selected community levels
    :return: Markov_time_ls is the list of markov times corresponding to selected community levels,
        starting from finest scale preset at markov time 0, ending at coarsest scale (1 cluster)
    """

    L_allgenes_Markov_time_ls = get_Markov_time_ls(L_all_results,L_comm_levels)
    R_allgenes_Markov_time_ls = get_Markov_time_ls(R_all_results,R_comm_levels)
    LR_exp_in_clusters = [np.concatenate((L, R), axis=0) for L, R in zip(L_exp_in_clusters, R_exp_in_clusters)]
    data = LR_exp_in_clusters

    def n(a):
        return a / np.sum(a)

    def metric(exp1, exp2):
        # exp1: pathway 1's ligand expression in each cluster of L_comm_levels + ...receptor...
        L1 = exp1[:np.sum(L_comm_levels)]
        L2 = exp2[:np.sum(L_comm_levels)]
        R1 = exp1[np.sum(L_comm_levels):]
        R2 = exp2[np.sum(L_comm_levels):]

        L_ind_split = [int(np.sum(L_comm_levels[:i])) for i in range(1, len(L_comm_levels) + 1)]
        R_ind_split = [int(np.sum(R_comm_levels[:i])) for i in range(1, len(R_comm_levels) + 1)]
        L_weights = np.asarray([1 / (1 + a) for a in L_allgenes_Markov_time_ls])
        R_weights = np.asarray([1 / (1 + a) for a in R_allgenes_Markov_time_ls])

        L_cosine = np.asarray(
            [scipy.spatial.distance.cosine(a, b) for a, b in zip(np.split(L1, L_ind_split), np.split(L2, L_ind_split))
             if len(a) > 0])
        R_cosine = np.asarray(
            [scipy.spatial.distance.cosine(a, b) for a, b in zip(np.split(R1, R_ind_split), np.split(R2, R_ind_split))
             if len(a) > 0])
        return np.dot(L_weights, L_cosine) + np.dot(R_weights, R_cosine)

    fit = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric=metric
    )

    umap_coord = fit.fit_transform(data)
    kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(umap_coord)

    return umap_coord,kmeans

import seaborn as sns
import matplotlib.pyplot as plt
# Draw jaccard similarity between multiscale cell clusters and cell type annotations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

def jaccard_dist(comm_ids,celltype_annotations,save=False):
    celltypes = np.unique(celltype_annotations)
    jaccard_dist_mtx = []
    for comm_id in comm_ids:
        for cluster in range(int(np.max(comm_id) + 1)):
            jaccard_dist_row = []
            for j, celltype in enumerate(celltypes):
                u = comm_id == cluster
                v = (celltype_annotations == celltype)
                jaccard_dist_row.append(scipy.spatial.distance.jaccard(u, v))
            jaccard_dist_mtx.append(jaccard_dist_row)
    jaccard_dist_mtx = np.asarray(jaccard_dist_mtx)
    ax = sns.heatmap(jaccard_dist_mtx, linewidth=0.5)
    ticks = np.arange(len(celltypes))+0.5
    plt.xticks(ticks=ticks, labels=celltypes)
    if save:
        plt.savefig(f'./figures/jaccard_dist_celltype.pdf')
    plt.show()

def get_new_lr_sup(ligand_sup, receptor_sup, spatial_dist_mtx):
    new_ligand_sup = np.matmul(ligand_sup.reshape((-1, 1)) * spatial_dist_mtx, receptor_sup.reshape((-1, 1)))
    new_receptor_sup = np.matmul(receptor_sup.reshape((-1, 1)) * spatial_dist_mtx, ligand_sup.reshape((-1, 1)))
    return [new_ligand_sup, new_receptor_sup]

def obtain_spatial_pca(pca_embedding,spatial_pos,w=0.5):
#w: weight of spatial importance
    row_norms = np.linalg.norm(pca_embedding, axis=1)
    scaled_pca = pca_embedding / np.mean(row_norms)
    shifted_pos = spatial_pos - np.mean(spatial_pos, axis=0)
    shifted_pos_row_norms = np.linalg.norm(shifted_pos, axis=1)
    scaled_pos = shifted_pos / np.mean(shifted_pos_row_norms)
    spatial_pca = np.concatenate((scaled_pca, w * scaled_pos), axis=1)

    return spatial_pca
