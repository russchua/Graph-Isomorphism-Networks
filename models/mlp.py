import tensorflow as tf

class MLP(tf.keras.layers.Layer):
  def __init__(self,num_layers, hidden_dim, output_dim):
    '''
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
    '''    
    super(MLP,self).__init__()
    
    self.linear_or_not = True #default is linear model
    self.num_layers = num_layers
    
    if num_layers < 1:
        raise ValueError("number of layers should be positive!")
    elif num_layers == 1:
        #Linear model
        self.linear = Linear_model(output_dim = output_dim)
    else:
        #Multi-layer model
        self.linear_or_not = False
        self.multi = Multi_model(layers = num_layers,hidden_dim = hidden_dim,output_dim = output_dim)
                     
  def call(self,input_features):
    if self.linear_or_not:
        #If linear model
        return self.linear(input_features)
    else:
        #If MLP
        return self.multi(input_features)
    
  
class Linear_model(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(Linear_model,self).__init__()
    self.output_layer = tf.keras.layers.Dense(units = output_dim)
    
  def call(self,input_features):
    return self.output_layer(input_features)  
    
  
class Multi_model(tf.keras.layers.Layer):
  def __init__(self,layers,hidden_dim,output_dim):
    super(Multi_model,self).__init__()
    self.layers = layers
    self.dense_list = []
    self.batch_list = []
    
    for i in range(layers-1):
      self.dense_list.append(tf.keras.layers.Dense(units = hidden_dim))
      self.batch_list.append(tf.keras.layers.BatchNormalization())
    self.dense_list.append(tf.keras.layers.Dense(units = output_dim))
    
  def call(self,input_features):
    for i in range(self.layers-1):
      densed = self.dense_list[i](input_features)
      batched = self.batch_list[i](densed)
      input_features = tf.nn.relu(batched)
    multi_result = self.dense_list[-1](input_features)
  
    return multi_result 
  
