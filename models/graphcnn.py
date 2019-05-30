import tensorflow as tf
import sys
sys.path.append("models/")
from mlp import MLP

class GraphCNN(tf.keras.Model):
    def __init__(self, num_layers, num_mlp_layers, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = tf.Variable(tf.zeros(self.num_layers-1))

    
        ###List of MLPs
        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.mlps = []
        self.batches = []
        self.linears = []
        self.drops = []
        
        for layer in range(self.num_layers-1):
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim))
            self.batches.append(tf.keras.layers.BatchNormalization())
            self.linears.append(tf.keras.layers.Dense(output_dim))
            self.drops.append(tf.keras.layers.Dropout(final_dropout))
                    
        self.linears.append(tf.keras.layers.Dense(output_dim))
        self.drops.append(tf.keras.layers.Dropout(final_dropout))
                
    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph
        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])
        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])
                
                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)        
        return tf.constant(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = tf.concat(edge_mat_list,1)
        Adj_block_elem = tf.ones(Adj_block_idx.shape[1])
        
        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
        if not self.learn_eps:
            num_node = start_idx[-1]  #This is the number of nodes in the entire graph            
            self_loop_edge = tf.constant([range(num_node),range(num_node)])
            elem = tf.ones(num_node)
            Adj_block_idx = tf.concat([Adj_block_idx, self_loop_edge], 1)   #Adding connections from self-connections to the list of other connections specified
            Adj_block_elem = tf.concat([Adj_block_elem, elem], 0)  #Total number of connections + number of nodes
            
        Adj_block_idx = tf.cast(tf.transpose(Adj_block_idx),tf.int64)
        Adj_block = tf.SparseTensor(indices=Adj_block_idx, values=Adj_block_elem, dense_shape=[start_idx[-1],start_idx[-1]])        
        return Adj_block

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)        
        start_idx = [0]
        
        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            #By default, it goes to the else
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])  #idx will be [[0,0],[0,1]...[0,218<end of the 1st graph>],[1,219<end of the first graph + 1>]...]
        
        elem = tf.constant(elem)        
        graph_pool = tf.SparseTensor(indices=idx, values=elem, dense_shape=[len(batch_graph),start_idx[-1]])
        graph_pool = tf.cast(graph_pool,tf.float32)        #graph_pool is a diagonal matrix of ones, where the ones are rows corresponding to the length of each graph.
        return graph_pool
        
    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling
        dummy = tf.reduce_min(tf_output,axis=0,keepdims=True)
        h_with_dummy = tf.concat([tf_output,tf_dummy],0)        
        pooled_rep = tf.reduce_max(h_with_dummy[padded_neighbor_list], axis = 1)        
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            h2 = tf.cast(h,tf.float32)
            pooled = tf.sparse.sparse_dense_matmul(Adj_block,h2)

            if self.neighbor_pooling_type == "average":  #The default is sum
                #If average pooling                
                degree = tf.sparse.sparse_dense_matmul(Adj_block,tf.ones([Adj_block.shape[0],1]))
                pooled = pooled/degree
       
        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h2        
        pooled_rep = self.mlps[layer](pooled)
                
        h = self.batches[layer](pooled_rep)        
        h = tf.nn.relu(h)
        h = tf.cast(h,tf.float32)                
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling            
            pooled = tf.sparse.sparse_dense_matmul(Adj_block,h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = tf.sparse.sparse_dense_matmul(Adj_block,tf.ones([Adj_block.shape[0],1]))                
                pooled = pooled/degree

        #representation of neighboring and center nodes        
        pooled_rep = self.mlps[layer](pooled)
        
        h = self.batches[layer](pooled_rep)
        
        #non-linearity
        h = tf.nn.relu(h)        
        return h
      
    
    def call(self, batch_graph):                
        X_concat = tf.concat([graph.node_features for graph in batch_graph],0)        
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)
        
        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat
                
        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:  #This is the one that triggers for the reddit dataset
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)        
            hidden_rep.append(h)
        score_over_layer = 0
    
        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):        
            h = tf.cast(h,tf.float32)
            pooled_h = tf.sparse.sparse_dense_matmul(graph_pool,h)            
            linear_outcome = self.linears[layer](pooled_h)
            dropped_outcome = self.drops[layer](linear_outcome)
            score_over_layer += dropped_outcome

        return score_over_layer  #This actually provides the value for the prediction      
