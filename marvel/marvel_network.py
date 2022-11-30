import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import for the model
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.layout import bipartite_layout

def loadData():
    # Data loading
    edges_raw = pd.read_csv("edges.csv")
    nodes_raw = pd.read_csv("nodes.csv")
    heros_raw = pd.read_csv("hero-network.csv")
    return (edges_raw, nodes_raw , heros_raw )

def miniBipartiteModel():
    # Mini sample
    mini_comic_list = ['TB:LS', 'ASM 53', 'MX 16', 'GAL 3', 'COH 2']
    mini_hero_list = ['CAPTAIN MARVEL III/G', 'JAMESON, J. JONAH', 'BOUDREAUX, BELLA DON', 
                  'SCARLET WITCH/WANDA', 'STORM/ORORO MUNROE S']
    mini_edges = [('CAPTAIN MARVEL III/G', 'TB:LS'),
              ('JAMESON, J. JONAH', 'ASM 53'),
              ('BOUDREAUX, BELLA DON', 'MX 16'),
              ('SCARLET WITCH/WANDA', 'GAL 3'),
              ('STORM/ORORO MUNROE S', 'COH 2'),
              ('JAMESON, J. JONAH', 'GAL 3'),
              ('STORM/ORORO MUNROE S', 'TB:LS'),
              ('JAMESON, J. JONAH', 'MX 16'),
             ]

# Make a blank graph
    mini_bipart = nx.Graph()

# Add nodes
    mini_bipart.add_nodes_from(mini_comic_list, bipartite=0)
    mini_bipart.add_nodes_from(mini_hero_list, bipartite=1)

# Add edges
    mini_bipart.add_edges_from(mini_edges)

# Separate nodes
    mini_top_nodes = {n for n, d in mini_bipart.nodes(data=True) if d['bipartite'] == 0}
    mini_bottom_nodes = set(mini_bipart) - mini_top_nodes
    mini_bipart.nodes(data=True)

# Set node colors
    color_dict = {0:'cornflowerblue', 1:'tomato'}
    color_list = [color_dict[i[1]] for i in mini_bipart.nodes.data('bipartite')]

# Draw a bipartite graph
    pos = dict()
    color = []
    pos.update((n, (1, i)) for i, n in enumerate(mini_bottom_nodes) ) 
    pos.update((n, (2, i)) for i, n in enumerate(mini_top_nodes) ) 
    nx.draw(mini_bipart, pos=pos, with_labels=True, node_color=color_list, font_size=8)
    plt.show()
    plt.savefig("mini_bipartite_graph.png")


def bipartiteModel(edges_raw, nodes_raw , heros_raw ):
    # Full bipartite graph

# Separate comic and hero nodes
    comic_node = nodes_raw['node'][nodes_raw['type'] == 'comic'].unique()
    hero_node = nodes_raw['node'][nodes_raw['type'] == 'hero'].unique()

# Make the edge data as a 2-tuple
    edges_tuple = edges_raw.values.tolist()
    edges_tuple = [tuple(x) for x in edges_tuple]

# Make numpy array to list
    comic_node_list = comic_node.tolist()
    hero_node_list = hero_node.tolist()

# Make a blank graph
    bipart = nx.Graph()

# Add nodes
    bipart.add_nodes_from(comic_node_list, bipartite=0)
    bipart.add_nodes_from(hero_node_list, bipartite=1)
    bipart.add_node("SPIDER-MAN/PETER PARKER", bipartite=1) # Manually added a node that was somehow not included 

# Add edges
    bipart.add_edges_from(edges_tuple)

# Separate nodes
    top_nodes = {n for n, d in bipart.nodes(data=True) if d['bipartite'] == 0}
    bottom_nodes = set(bipart) - top_nodes

# Visualization
    plt.figure(figsize=(70, 6))
    pos = bipartite_layout(bipart, top_nodes, align='horizontal')
    nx.draw(bipart, pos=pos, node_size=10, node_color='lightgreen', alpha=0.01)
    nx.draw_networkx_labels(bipart, pos=pos, font_size=3)
    plt.show()
    plt.savefig("full_bipartite_graph.png")
    return (bipart, comic_node_list , hero_node_list)
    
def heroUnipartiteModel(bipart, hero_node_list):
        # Build a unipartite graph of hero
    hero_graph = bipartite.projected_graph(bipart, hero_node_list, multigraph=False)

# Visualization
    plt.figure(figsize=(5, 5))
    pos=nx.spring_layout(hero_graph)
    nx.draw(hero_graph, pos=pos, node_size=5, node_color='tomato', alpha=0.4)
    nx.draw_networkx_edges(hero_graph, pos=pos, alpha=0.1)
    plt.show()
    plt.savefig("hero_unipartite_graph.png")

def comicsUnipartiteModel(bipart, comic_node_list):
# Build a unipartite graph of comic
    comic_graph = bipartite.projected_graph(bipart, comic_node_list, multigraph=False)

# Visualization
    plt.figure(figsize=(5, 5))
    pos=nx.spring_layout(comic_graph)
    nx.draw(comic_graph, pos=pos, node_size=5, node_color='cornflowerblue', alpha=0.4)
    nx.draw_networkx_edges(comic_graph, pos=pos, alpha=0.1)
    plt.show()
    plt.savefig("commics_unipartite_graph.png")


def main():
    edges_raw, nodes_raw , heros_raw  = loadData()
    miniBipartiteModel()
    bipart, comic_node_list , hero_node_list= bipartiteModel(edges_raw, nodes_raw , heros_raw )
    heroUnipartiteModel(bipart, hero_node_list)
    comicsUnipartiteModel(bipart, comic_node_list)
        
if __name__ == '__main__':
    main()