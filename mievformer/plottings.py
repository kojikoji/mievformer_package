# import squidpy as sq
from sklearn.neighbors import NearestNeighbors
from matplotlib import colors, cm
from matplotlib import pyplot as plt
# from scripts import utils
import numpy as np
import networkx as nx

## based on https://stackoverflow.com/questions/14938541/how-to-improve-the-label-placement-in-scatter-plot
def repel_labels(ax, xs, ys, labs, k=0.01):
    G = nx.DiGraph()
    data_ids = [f'data_{lab}' for lab in labs]
    for lab, data_id in zip(labs, data_ids):
        G.add_node(lab)
        G.add_node(data_id)
        G.add_edge(lab, data_id)
    init_pos_dict = {
        key: (x, y)
        for key, x, y in zip(
            np.concatenate([labs, data_ids]),
            np.tile(xs, 2), np.tile(ys, 2))
    }
    mod_pos_dict = nx.spring_layout(G, pos=init_pos_dict, fixed=data_ids, k=k, scale=None)
    arrowprops = dict(arrowstyle='->', shrinkA=0, shrinkB=0, connectionstyle='arc3')
    for lab, data_id in G.edges():
        ax.annotate(lab, xy=mod_pos_dict[data_id], xytext=mod_pos_dict[lab], xycoords='data', textcoords='data', arrowprops=arrowprops, fontsize=12)
    
def annotated_scatters(ax, xs, ys, sub_idxs, labs=None, k=0.01):
    ax.scatter(xs, ys, s=1, c='gray')
    if sub_idxs.shape[0] > 0:
        ax.scatter(xs[sub_idxs], ys[sub_idxs], s=10, c='red')
        if not labs is None:
            sub_labs = labs[sub_idxs]
        else:
            sub_labs = sub_idxs
        repel_labels(ax, xs[sub_idxs], ys[sub_idxs], sub_labs, k=k)


def annotated_bars(ax, labels, values, color='gray'):
    xs = np.arange(len(values))
    ax.bar(xs, values, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right')

def annotated_hbars(ax, labels, values):
    ys = np.arange(len(labels))
    ax.barh(ys, values)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

def visualize_volcano(ax, deg_df, top_genes, k=0.3):
    deg_df = deg_df.copy()
    deg_df.index = deg_df.names
    annotated_scatters(ax, deg_df['logfoldchanges'], -np.log10(deg_df['pvals_adj']), top_genes, k=k)
    ax.set_xlabel('log2FoldChange')
    ax.set_ylabel('-log10(p-value adjusted)')
