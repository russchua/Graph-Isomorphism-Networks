import networkx as nx
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    triggered = False

    
    #This 1st section of the load_data function reads every row in the text file.
    #The first row of every graph will provide you with 2 things, the number of nodes, the associated label
    #The subsequent rows represent each node, and will begin with 0
    #Represented in this order (0,number of connections, <nodes that they are connected to>)
    #The graphs are created to store each node, and to add the edge lists inside
    #Thereafter, each graph is stored into a list known as g_list
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    triggered = True
                if triggered:
                  print('Triggered already')
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags))

            
    #This is the 2nd part of the load_data code.
    #Within each graph class, he is creating the following:
    #1. graph.neighbors <matrix> : This is similar to the matrix A, but it only contains the data of the connections.
    #2. graph.max_neighbor <int>: This is basically the maximum number of connections that arises from a single node in the graph
    #3. graph.label <int>        : This is to ensure that the labels do not skip numbers, i.e. labels <1,2,4,5> becomes <0,1,2,3>
    #4. graph.edge_mat <matrix>  : This is a transpose of the edge matrix from the graph [[nodes][connected nodes]]
            
    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
            
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]  #This line appears to be redundant
            degree_list.append(len(g.neighbors[i]))         
        g.max_neighbor = max(degree_list)
        g.label = label_dict[g.label]  #All this line does, is ensure that there is no skipping of labels. The actual numerical label is just a variable.
    
        #These two lines create an extension of the edges to ensure that they are bi-directional
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        
        g.edge_mat = tf.transpose(tf.constant(edges))
        
        
        
    #This part gives the degree dictionary but not in the order of nodes within the dataset.txt file  
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    
    
    for g in g_list:
        
        node_features = np.zeros((len(g.node_tags), len(tagset)))
        node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1  
        g.node_features = tf.constant(node_features)
        
        
    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    
    train_idx, test_idx = idx_list[fold_idx]
    
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list



