import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from mpl_toolkits.mplot3d.proj3d import proj_transform
from wordcloud import WordCloud

from .Data_preparation import get_pathway_genes,get_gene_ind_in_gene_ls,find_lr_intersected_clusters,get_rel_i_j_ls,\
    get_CCC_mtx, get_cluster_size_ls, get_general_tree_colors

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """
    Add an 3d arrow to an `Axes3D` instance.
    """

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

def build_tree_from_rel(rel):
    n_nodes = np.max(rel[0]) + 1
    for r in rel:
        n_nodes += len(r)
    adj_mtx = np.zeros((n_nodes, n_nodes))
    slide_x = 0
    slide_y = 0
    for r in rel:
        slide_y += np.max(r) + 1
        for child_level, parent_level in enumerate(r):
            adj_mtx[parent_level + slide_x, child_level + slide_y] = 1
        slide_x = slide_y

    tree = nx.DiGraph()
    tree.add_nodes_from(range(n_nodes))
    tree.add_edges_from(np.transpose(np.nonzero(adj_mtx)))
    node_pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')

    tree.add_node(n_nodes)
    tree.add_edges_from([(n_nodes,i) for i in range(np.max(rel[0])+1)])

    ratio = len(rel) / (len(rel) + 1)
    top_x = 0
    top_y = 0

    for node in range(n_nodes):
        x, y = node_pos[node]

        if node < np.max(rel[0]) + 1:
            top_x += x / (np.max(rel[0]) +1)
            top_y = y

        y = y * ratio
        node_pos[node] = (x, y)

    node_pos[n_nodes] = (top_x, top_y)

    return tree, node_pos

def get_node_3D_pos(lr, node_pos):
    node_3D_pos = {}

    if lr == 'L':
        plane_level = 0
    else:
        plane_level = 1

    x_max = 0
    y_max = 0

    for node in node_pos.keys():
        x, y = node_pos[node]
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    for node in node_pos.keys():
        x, y = node_pos[node]
        node_3D_pos[node] = (2 * x / x_max, plane_level, y / y_max)

    return node_3D_pos

def draw_CCC(L_onehot_ls, R_onehot_ls, CCC_mtx, L_node_mapping_dict=None, R_node_mapping_dict=None, L_is_general=True, R_is_general=True,CCC_linewidth=10,CCC_nodesize=5,save=None,angle=-25,height_angle=10):
    """
    draw_CCC draws CCC of 4 cases: 1) allgenes to allgenes 2)L-general to R-general 3)L-MMT to R-general 4)L-general to R-MMT

    :param: all_LR: LR interactions from CellChatDB
    :param: complex_input: complex_input file from CellChatDB
    :param: lr: 'L' or 'R'
    :param: ds_mtx: gene expression matrix
    :param: ds_geneNames: dataframe of gene names
    :param: angle: horizontal angle of view
    :param: height_angle: height angle of view
    :return: lr_sup_mtx is the support matrix of ligands/receptors,
        all_LR_lr_ind is dataframe of list of indices of ligands/receptors in all_LR,
        lr_ls is list of unique ligands/receptors, in which ligand/receptor complex uses its complex name
        separate_lr is list of unique ligands/receptors, in which ligand/receptor complex separates into different gene names
    """
    w = 16
    h = 16

    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=300, subplot_kw={'projection': '3d'})

    xx, zz = np.meshgrid([-0.2, 2.2], [-0.1, 1.1])

    ax.plot_surface(xx, 0, zz, color='#AFEEEE', alpha=0.1, zorder=0)
    ax.plot_surface(xx, 1, zz, color='#AFEEEE', alpha=0.1, zorder=0)

    L_size_ls = get_cluster_size_ls(L_onehot_ls)
    R_size_ls = get_cluster_size_ls(R_onehot_ls)

    L_color_ls = sum(get_general_tree_colors(L_onehot_ls),[])
    R_color_ls = sum(get_general_tree_colors(R_onehot_ls),[])

    L_size_ls = sum(L_size_ls, [])
    R_size_ls = sum(R_size_ls, [])

    L_rel_i_j = get_rel_i_j_ls(L_onehot_ls)[0]
    L_tree, L_node_pos = build_tree_from_rel(L_rel_i_j)
    R_rel_i_j = get_rel_i_j_ls(R_onehot_ls)[0]
    R_tree, R_node_pos = build_tree_from_rel(R_rel_i_j)

    L_node_3D_pos = get_node_3D_pos('L', L_node_pos)
    R_node_3D_pos = get_node_3D_pos('R', R_node_pos)

    if L_is_general and R_is_general:
        for i in range(len(L_node_3D_pos) - 1):
            x, y, z = L_node_3D_pos[i]
            ax.scatter(x, y, z, s=CCC_nodesize*1000 * L_size_ls[i]**(1/3), c=L_color_ls[i])
            ax.text(x,y,z-0.1, f'{i}',c='black')

        for i in range(len(R_node_3D_pos) - 1):
            x, y, z = R_node_3D_pos[i]
            ax.scatter(x, y, z, s=CCC_nodesize*1000 * R_size_ls[i]**(1/3), c=R_color_ls[i])
            ax.text(x, y, z - 0.1, f'{i}', c='black')

        # filter CCC between duplicate nodes
        L_filter = np.asarray([len(L_tree.out_edges(i)) != 1 for i in range(len(L_color_ls))]) # filter out nodes that has only one child identical to it
        R_filter = np.asarray([len(R_tree.out_edges(i)) != 1 for i in range(len(R_color_ls))]) # filter out nodes that has only one child identical to it

        LR_filter = np.zeros((len(L_filter), len(R_filter)))
        for i in range(len(L_filter)):
            for j in range(len(R_filter)):
                if L_filter[i] and R_filter[j]:
                    LR_filter[i][j] = 1

        CCC_mtx = np.multiply(LR_filter, CCC_mtx) # filter CCC mtx

        for i in range(len(L_node_3D_pos) - 1):
            lx, ly, lz = L_node_3D_pos[i]
            for j in range(len(R_node_3D_pos) - 1):
                rx, ry, rz = R_node_3D_pos[j]
                if CCC_mtx[i][j] > 0:
                    ax.arrow3D(lx, ly, lz, rx - lx, ry - ly, rz - lz, linewidth=CCC_linewidth*CCC_mtx[i, j], arrowstyle='-', fc=L_color_ls[i], ec=L_color_ls[i])
                    ax.arrow3D((lx+rx)/2, (ly+ry)/2, (lz+rz)/2, (rx - lx)/100, (ry - ly)/100, (rz - lz)/100, linewidth=2, mutation_scale=5,
                               fc=L_color_ls[i], ec=L_color_ls[i], arrowstyle=
                               ArrowStyle.CurveB(head_length=2, head_width=1))

    if not L_is_general and R_is_general:
        assert L_node_mapping_dict is not None
        for node in L_node_3D_pos.keys():
            x, y, z = L_node_3D_pos[node]
            ax.scatter(x, y, z, s=CCC_nodesize*1000*L_size_ls[L_node_mapping_dict[node]]**(1/3), c=L_color_ls[L_node_mapping_dict[node]])

        for i in range(len(R_node_3D_pos) - 1):
            x, y, z = R_node_3D_pos[i]
            ax.scatter(x, y, z, s=CCC_nodesize*1000*R_size_ls[i]**(1/3), c=R_color_ls[R_node_mapping_dict[node]])

        for node in L_node_3D_pos.keys():
            lx, ly, lz = L_node_3D_pos[node]
            i = L_node_mapping_dict[node]
            for j in range(len(R_node_3D_pos) - 1):
                rx, ry, rz = R_node_3D_pos[j]
                if CCC_mtx[i][j] > 0:
                    ax.arrow3D(lx, ly, lz, rx - lx, ry - ly, rz - lz, linewidth=CCC_linewidth*CCC_mtx[i, j], arrowstyle='-', fc=L_color_ls[i], ec=L_color_ls[i])
                    ax.arrow3D((lx+rx)/2, (ly+ry)/2, (lz+rz)/2, (rx - lx)/100, (ry - ly)/100, (rz - lz)/100, linewidth=2, mutation_scale=5,
                               fc=L_color_ls[i], ec=L_color_ls[i], arrowstyle=
                               ArrowStyle.CurveB(head_length=2, head_width=1))

    if L_is_general and not R_is_general:
        assert R_node_mapping_dict is not None
        for node in R_node_3D_pos.keys():
            x, y, z = R_node_3D_pos[node]
            ax.scatter(x, y, z, s=CCC_nodesize*1000*R_size_ls[R_node_mapping_dict[node]]**(1/3),c=R_color_ls[R_node_mapping_dict[node]])

        for i in range(len(L_node_3D_pos) - 1):
            x, y, z = L_node_3D_pos[i]
            ax.scatter(x, y, z, s=CCC_nodesize*1000*L_size_ls[i]**(1/3),c=L_color_ls[i])

        for node in R_node_3D_pos.keys():
            rx, ry, rz = R_node_3D_pos[node]
            i = R_node_mapping_dict[node]
            for j in range(len(L_node_3D_pos) - 1):
                lx, ly, lz = L_node_3D_pos[j]
                if CCC_mtx[j][i] > 0:
                    ax.arrow3D(lx, ly, lz, rx - lx, ry - ly, rz - lz, linewidth=CCC_linewidth*CCC_mtx[j, i], arrowstyle='-', fc=L_color_ls[j], ec=L_color_ls[j])
                    ax.arrow3D((lx+rx)/2, (ly+ry)/2, (lz+rz)/2, (rx - lx)/100, (ry - ly)/100, (rz - lz)/100, linewidth=2, mutation_scale=5,
                               fc=L_color_ls[j], ec=L_color_ls[j], arrowstyle=ArrowStyle.CurveB(head_length=2, head_width=1))

    for l1, l2 in L_tree.edges:
        l1x, l1y, l1z = L_node_3D_pos[l1]
        l2x, l2y, l2z = L_node_3D_pos[l2]
        ax.arrow3D(l1x, l1y, l1z, l2x - l1x, l2y - l1y, l2z - l1z, linewidth=0.5, color='b')

    for r1, r2 in R_tree.edges:
        r1x, r1y, r1z = R_node_3D_pos[r1]
        r2x, r2y, r2z = R_node_3D_pos[r2]
        ax.arrow3D(r1x, r1y, r1z, r2x - r1x, r2y - r1y, r2z - r1z, linewidth=0.5, color='g')

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)

    ax.text(0, 0, 1, "Ligand",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')
    ax.text(0, 1, 1, "Receptor",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')

    # select viewing angle
    ax.view_init(height_angle, angle)

    # how much do you want to zoom into the fig
    ax.dist = 10
    ax.set_axis_off()
    if save is not None:
        plt.savefig(f'./figures/{save}', dpi=425, bbox_inches='tight', transparent=True)
    plt.show()

def draw_intersected_CCC(pathway,all_LR,L_ls,R_ls,L_onehot_ls,R_onehot_ls,L_sup_mtx,R_sup_mtx,
                     ligand_exp_dict,receptor_exp_dict,cofactor_exp_dict,CCC_threshold=0.5,CCC_linewidth=10,CCC_nodesize=5,save=None):
    """
    draw_CCC draws CCC of 4 cases: 1) allgenes to allgenes 2)L-general to R-general 3)L-MMT to R-general 4)L-general to R-MMT

    :param: all_LR: LR interactions from CellChatDB
    :param: complex_input: complex_input file from CellChatDB
    :param: lr: 'L' or 'R'
    :param: ds_mtx: gene expression matrix
    :param: ds_geneNames: dataframe of gene names
    :return: lr_sup_mtx is the support matrix of ligands/receptors,
        all_LR_lr_ind is dataframe of list of indices of ligands/receptors in all_LR,
        lr_ls is list of unique ligands/receptors, in which ligand/receptor complex uses its complex name
        separate_lr is list of unique ligands/receptors, in which ligand/receptor complex separates into different gene names
    """
    w = 16
    h = 16

    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=300, subplot_kw={'projection': '3d'})

    xx, zz = np.meshgrid([-0.2, 2.2], [-0.1, 1.1])

    ax.plot_surface(xx, 0, zz, color='#AFEEEE', alpha=0.1, zorder=0)
    ax.plot_surface(xx, 1, zz, color='#AFEEEE', alpha=0.1, zorder=0)

    L_original_color_ls = sum(get_general_tree_colors(L_onehot_ls),[])
    R_original_color_ls = sum(get_general_tree_colors(R_onehot_ls),[])

    def L_color_shape_map(onehot, is_contain):
        ind = np.where(np.sum(np.concatenate(L_onehot_ls, axis=0) - onehot, axis=1) == 0)[0][0]
        if is_contain == 1:
            return L_original_color_ls[ind], '^'
        else:
            return L_original_color_ls[ind], 'o'

    def R_color_shape_map(onehot, is_contain):
        ind = np.where(np.sum(np.concatenate(R_onehot_ls, axis=0) - onehot, axis=1) == 0)[0][0]
        if is_contain == 1:
            return R_original_color_ls[ind], '^'
        else:
            return R_original_color_ls[ind], 'o'

    pathway_ligands, pathway_receptors = get_pathway_genes(all_LR, pathway)
    pathway_ligands_ind = get_gene_ind_in_gene_ls(pathway_ligands, L_ls)
    pathway_receptors_ind = get_gene_ind_in_gene_ls(pathway_receptors, R_ls)

    L_intersected_onehot_ls, L_original_onehot_ls = find_lr_intersected_clusters(pathway_ligands_ind, L_onehot_ls, L_sup_mtx)
    R_intersected_onehot_ls, R_original_onehot_ls = find_lr_intersected_clusters(pathway_receptors_ind, R_onehot_ls, R_sup_mtx)

    L_intersected_color_ls = [L_color_shape_map(onehot, is_contain)[0] for onehot, is_contain in sum(L_original_onehot_ls, [])]
    L_intersected_shape_ls = [L_color_shape_map(onehot, is_contain)[1] for onehot, is_contain in sum(L_original_onehot_ls, [])]

    R_intersected_color_ls = [R_color_shape_map(onehot, is_contain)[0] for onehot, is_contain in sum(R_original_onehot_ls, [])]
    R_intersected_shape_ls = [R_color_shape_map(onehot, is_contain)[1] for onehot, is_contain in sum(R_original_onehot_ls, [])]

    L_size_ls = get_cluster_size_ls(L_intersected_onehot_ls)
    R_size_ls = get_cluster_size_ls(R_intersected_onehot_ls)
    L_size_ls = sum(L_size_ls, [])
    R_size_ls = sum(R_size_ls, [])

    L_rel_i_j = get_rel_i_j_ls(L_intersected_onehot_ls)[0]
    L_tree, L_node_pos = build_tree_from_rel(L_rel_i_j)
    R_rel_i_j = get_rel_i_j_ls(R_intersected_onehot_ls)[0]
    R_tree, R_node_pos = build_tree_from_rel(R_rel_i_j)

    L_node_3D_pos = get_node_3D_pos('L', L_node_pos)
    R_node_3D_pos = get_node_3D_pos('R', R_node_pos)

    for i in range(len(L_node_3D_pos) - 1):
        x, y, z = L_node_3D_pos[i]
        ax.scatter(x, y, z, s= CCC_nodesize*1000*L_size_ls[i]**(1/2), c=L_intersected_color_ls[i], marker=L_intersected_shape_ls[i])

    for i in range(len(R_node_3D_pos) - 1):
        x, y, z = R_node_3D_pos[i]
        ax.scatter(x, y, z, s=CCC_nodesize*1000*R_size_ls[i]**(1/2), c=R_intersected_color_ls[i], marker=R_intersected_shape_ls[i])

    pathway_CCC_mtx = get_CCC_mtx(all_LR, pathway_ligands, pathway_receptors, L_intersected_onehot_ls,
                                  R_intersected_onehot_ls, ligand_exp_dict, receptor_exp_dict, cofactor_exp_dict,
                                  CCC_threshold)

    CCC_mtx = np.concatenate([np.concatenate(
        [pathway_CCC_mtx[(l_level, r_level)] for l_level in range(len(L_intersected_onehot_ls))], axis=0) for r_level in
        range(len(R_intersected_onehot_ls))], axis=1)

    # filter CCC between duplicate nodes
    L_filter = np.asarray([len(L_tree.out_edges(i)) != 1 for i in range(len(L_intersected_color_ls))]) # filter out nodes that has only one child identical to it
    R_filter = np.asarray([len(R_tree.out_edges(i)) != 1 for i in range(len(R_intersected_color_ls))]) # filter out nodes that has only one child identical to it

    LR_filter = np.zeros((len(L_filter), len(R_filter)))
    for i in range(len(L_filter)):
        for j in range(len(R_filter)):
            if L_filter[i] and R_filter[j]:
                LR_filter[i][j] = 1

    CCC_mtx = np.multiply(LR_filter, CCC_mtx) # filter CCC mtx

    for i in range(len(L_node_3D_pos) - 1):
        lx, ly, lz = L_node_3D_pos[i]
        for j in range(len(R_node_3D_pos) - 1):
            rx, ry, rz = R_node_3D_pos[j]
            if CCC_mtx[i][j] > 0:
                ax.arrow3D(lx, ly, lz, rx - lx, ry - ly, rz - lz, linewidth=CCC_linewidth*CCC_mtx[i, j], arrowstyle='-', fc=L_intersected_color_ls[i], ec=L_intersected_color_ls[i])
                ax.arrow3D((lx+rx)/2, (ly+ry)/2, (lz+rz)/2, (rx - lx)/100, (ry - ly)/100, (rz - lz)/100, linewidth=2, mutation_scale=5,
                           fc=L_intersected_color_ls[i], ec=L_intersected_color_ls[i], arrowstyle=
                           ArrowStyle.CurveB(head_length=2, head_width=1))

    for l1, l2 in L_tree.edges:
        l1x, l1y, l1z = L_node_3D_pos[l1]
        l2x, l2y, l2z = L_node_3D_pos[l2]
        ax.arrow3D(l1x, l1y, l1z, l2x - l1x, l2y - l1y, l2z - l1z, linewidth=0.5, color='b')

    for r1, r2 in R_tree.edges:
        r1x, r1y, r1z = R_node_3D_pos[r1]
        r2x, r2y, r2z = R_node_3D_pos[r2]
        ax.arrow3D(r1x, r1y, r1z, r2x - r1x, r2y - r1y, r2z - r1z, linewidth=0.5, color='g')

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)

    ax.text(0, 0, 1, "Ligand",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')
    ax.text(0, 1, 1, "Receptor",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')

    # select viewing angle
    angle = -25
    height_angle = 10
    ax.view_init(height_angle, angle)

    # how much do you want to zoom into the fig
    ax.dist = 10
    ax.set_axis_off()
    if save is not None:
        plt.savefig(f'./figures/{save}', dpi=425, bbox_inches='tight', transparent=True)
    plt.show()

def draw_MMT(lr_tree,lr_node_pos,lr_size_ls,lr_ls,lr_node_mapping_dict=None,nodesize=10,save=None):
    w = 16
    h = 16

    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=300)

    lr_size_ls = sum(lr_size_ls, [])
    node_3D_pos = get_node_3D_pos("L", lr_node_pos)
    for node in lr_node_pos.keys():
        x, _, y = node_3D_pos[node]
        ax.scatter(x/2, y, s=nodesize*1000*lr_size_ls[lr_node_mapping_dict[node]]**(1/3))
        ax.text(x/2-0.2*lr_size_ls[lr_node_mapping_dict[node]],y,lr_ls[node],fontsize=30)

    for l1, l2 in lr_tree.edges:
        l1x,_, l1y = node_3D_pos[l1]
        l2x,_, l2y = node_3D_pos[l2]
        ax.arrow(l1x/2, l1y, l2x/2 - l1x/2, l2y - l1y, linewidth=1, color='b')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    if save:
        plt.savefig(save)
    plt.show()

def draw_CCC_between_MMT(L_tree, L_node_pos, R_tree, R_node_pos, L_ls, L_size_ls, R_ls, R_size_ls, CCC_mtx, lr_pair_inds, L_node_mapping_dict=None, R_node_mapping_dict=None, CCC_linewidth = 10, CCC_nodesize=5,save=None):
    w = 16
    h = 16
    default_color_ls = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']

    L_color_ls = []
    R_color_ls = []
    for i, node in enumerate(L_tree.nodes()):
        L_color_ls.append(i % 10)
    for i, node in enumerate(R_tree.nodes()):
        R_color_ls.append(i % 10)

    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=300, subplot_kw={'projection': '3d'})

    xx, zz = np.meshgrid([-0.2, 2.2], [-0.1, 1.1])

    ax.plot_surface(xx, 0, zz, color='#AFEEEE', alpha=0.1, zorder=0)
    ax.plot_surface(xx, 1, zz, color='#AFEEEE', alpha=0.1, zorder=0)

    L_size_ls = sum(L_size_ls, [])
    R_size_ls = sum(R_size_ls, [])

    L_node_3D_pos = get_node_3D_pos('L', L_node_pos)
    R_node_3D_pos = get_node_3D_pos('R', R_node_pos)

    for node in L_node_3D_pos.keys():
        x, y, z = L_node_3D_pos[node]
        ax.scatter(x, y, z, s=CCC_nodesize*1000*L_size_ls[L_node_mapping_dict[node]]**(1/3), c=default_color_ls[L_color_ls[L_node_mapping_dict[node]]])
        ax.text(x,y,z-0.3*L_size_ls[L_node_mapping_dict[node]],L_ls[node],fontsize=15)

    for node in R_node_3D_pos.keys():
        x, y, z = R_node_3D_pos[node]
        ax.scatter(x, y, z, s=CCC_nodesize*1000*R_size_ls[R_node_mapping_dict[node]]**(1/3), c=default_color_ls[R_color_ls[R_node_mapping_dict[node]]])
        ax.text(x,y,z-0.3*R_size_ls[R_node_mapping_dict[node]],R_ls[node],fontsize=15)

    for node_i in L_node_3D_pos.keys():
        lx, ly, lz = L_node_3D_pos[node_i]
        i = L_node_mapping_dict[node_i]
        for node_j in R_node_3D_pos.keys():
            rx, ry, rz = R_node_3D_pos[node_j]
            j = R_node_mapping_dict[node_j]
            if CCC_mtx[i][j] > 0 and (node_i, node_j) in lr_pair_inds:
                ax.arrow3D(lx, ly, lz, rx - lx, ry - ly, rz - lz, fc=default_color_ls[L_color_ls[i]], ec=default_color_ls[L_color_ls[i]], linewidth=CCC_linewidth*CCC_mtx[i, j], arrowstyle='fancy')
                ax.arrow3D((lx+rx)/2, (ly+ry)/2, (lz+rz)/2, (rx - lx)/100, (ry - ly)/100, (rz - lz)/100, linewidth=2, mutation_scale=5,
                           fc=default_color_ls[L_color_ls[i]], ec=default_color_ls[L_color_ls[i]], arrowstyle=
                           ArrowStyle.CurveB(head_length=2, head_width=1))

    for l1, l2 in L_tree.edges:
        l1x, l1y, l1z = L_node_3D_pos[l1]
        l2x, l2y, l2z = L_node_3D_pos[l2]
        ax.arrow3D(l1x, l1y, l1z, l2x - l1x, l2y - l1y, l2z - l1z, linewidth=1, color='b')

    for r1, r2 in R_tree.edges:
        r1x, r1y, r1z = R_node_3D_pos[r1]
        r2x, r2y, r2z = R_node_3D_pos[r2]
        ax.arrow3D(r1x, r1y, r1z, r2x - r1x, r2y - r1y, r2z - r1z, linewidth=1, color='g')

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)

    ax.text(0, 0, 1, "Ligand",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')
    ax.text(0, 1, 1, "Receptor",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')

    # select viewing angle
    angle = -25
    height_angle = 10
    ax.view_init(height_angle, angle)

    # how much do you want to zoom into the fig
    ax.dist = 10
    ax.set_axis_off()
    if save is not None:
        plt.savefig(save, dpi=425, bbox_inches='tight', transparent=True)
    plt.show()

def draw_CCC_between_LR_union_MMT(LR_union_tree, LR_union_node_pos, all_LR_filtered, L_size_ls, R_size_ls, L_color_ls, R_color_ls, CCC_vector, LR_union_node_mapping_dict=None, CCC_linewidth=10,CCC_nodesize=5,save=None):

    w = 16
    h = 16
    default_color_ls = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=300, subplot_kw={'projection': '3d'})

    xx, zz = np.meshgrid([-0.2, 2.2], [-0.1, 1.1])

    ax.plot_surface(xx, 0, zz, color='#AFEEEE', alpha=0.1, zorder=0)
    ax.plot_surface(xx, 1, zz, color='#AFEEEE', alpha=0.1, zorder=0)

    L_node_3D_pos = get_node_3D_pos('L', LR_union_node_pos)
    R_node_3D_pos = get_node_3D_pos('R', LR_union_node_pos)

    for node in L_node_3D_pos.keys():
        x, y, z = L_node_3D_pos[node]
        ax.scatter(x, y, z, s=CCC_nodesize*1000*L_size_ls[LR_union_node_mapping_dict[node]]**(1/3), c=default_color_ls[L_color_ls[LR_union_node_mapping_dict[node]]])
        ax.text(x, y, z - 0.3 * L_size_ls[LR_union_node_mapping_dict[node]], all_LR_filtered.iloc[node]['Ligand'], fontsize=15, zorder=0)

    for node in R_node_3D_pos.keys():
        x, y, z = R_node_3D_pos[node]
        ax.scatter(x, y, z, s=CCC_nodesize*1000*R_size_ls[LR_union_node_mapping_dict[node]]**(1/3), c=default_color_ls[R_color_ls[LR_union_node_mapping_dict[node]]])
        ax.text(x, y, z - 0.3 * R_size_ls[LR_union_node_mapping_dict[node]], all_LR_filtered.iloc[node]['Receptor'], fontsize=15, zorder=0)

    for node_i in L_node_3D_pos.keys():
        lx, ly, lz = L_node_3D_pos[node_i]
        rx, ry, rz = R_node_3D_pos[node_i]
        i = LR_union_node_mapping_dict[node_i]

        # Draw interactions between 2 trees
        if CCC_vector[i] > 0:
            ax.arrow3D(lx, ly, lz, rx - lx, ry - ly, rz - lz, linewidth=CCC_linewidth*CCC_vector[i],
                       fc=default_color_ls[L_color_ls[i]], ec=default_color_ls[L_color_ls[i]], arrowstyle='-')
            ax.arrow3D((lx+rx)/2, (ly+ry)/2, (lz+rz)/2, (rx - lx)/100, (ry - ly)/100, (rz - lz)/100, linewidth=2,mutation_scale=5,
                       fc=default_color_ls[L_color_ls[i]], ec=default_color_ls[L_color_ls[i]], arrowstyle=
                       ArrowStyle.CurveB(head_length=3, head_width=2))

    # Draw edges within trees
    for l1, l2 in LR_union_tree.edges:
        l1x, l1y, l1z = L_node_3D_pos[l1]
        l2x, l2y, l2z = L_node_3D_pos[l2]
        ax.arrow3D(l1x, l1y, l1z, l2x - l1x, l2y - l1y, l2z - l1z, linewidth=1, color='b')

    for r1, r2 in LR_union_tree.edges:
        r1x, r1y, r1z = R_node_3D_pos[r1]
        r2x, r2y, r2z = R_node_3D_pos[r2]
        ax.arrow3D(r1x, r1y, r1z, r2x - r1x, r2y - r1y, r2z - r1z, linewidth=1, color='g')

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)

    ax.text(0, 0, 1, "Ligand",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')
    ax.text(0, 1, 1, "Receptor",
            color='black', fontsize='20', zorder=1e5, ha='left', va='center')

    # select viewing angle
    angle = -25
    height_angle = 10
    ax.view_init(height_angle, angle)

    # how much do you want to zoom into the fig
    ax.dist = 10
    ax.set_axis_off()
    if save is not None:
        plt.savefig(f'{save}', dpi=425, bbox_inches='tight', transparent=True)
    plt.show()

def is_leaf(node, edges):
    leafy_edges = set()
    for (x,y) in edges:
        if node == x:
            return False, None
        if node == y:
            leafy_edges.add((x,y))
    return True, leafy_edges

def is_root(node, edges):
    rooty_edges = set()
    for (x,y) in edges:
        if node == y:
            return False, None
        if node == x:
            rooty_edges.add((x,y))
    return True, rooty_edges

def get_node_level_bottom_up(tree):
    node_level_dict = {}
    remaining_nodes = set(tree.nodes)
    remaining_edges = set(tree.edges)

    level = 0
    total_leafy_edges = set()
    total_leveled_nodes = set()
    while remaining_nodes:
        node_level_dict[level] = []
        for node in list(remaining_nodes):
            leaf, leafy_edges = is_leaf(node, list(remaining_edges))
            if leaf:
                node_level_dict[level].append(node)
                total_leveled_nodes.add(node)
                total_leafy_edges = total_leafy_edges.union(leafy_edges)
        remaining_nodes -= total_leveled_nodes
        remaining_edges -= total_leafy_edges
        level += 1

    return node_level_dict

def get_node_level_top_down(tree):
    node_level_dict = {}
    remaining_nodes = set(tree.nodes)
    remaining_edges = set(tree.edges)

    level = 0
    total_rooty_edges = set()
    total_leveled_nodes = set()
    while remaining_nodes:
        node_level_dict[level] = []
        for node in list(remaining_nodes):
            root, rooty_edges = is_root(node, list(remaining_edges))
            if root:
                node_level_dict[level].append(node)
                total_leveled_nodes.add(node)
                total_rooty_edges = total_rooty_edges.union(rooty_edges)
        remaining_nodes -= total_leveled_nodes
        remaining_edges -= total_rooty_edges
        level += 1

    return node_level_dict

