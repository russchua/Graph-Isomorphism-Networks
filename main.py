import easydict
import numpy as np
from tqdm import tqdm

args = easydict.EasyDict({
    "dataset": 'MUTAG',
    "device": 0,
    "batch_size": 32,
    "iters_per_epoch": 50,
    "epochs": 350,#Change this back to 350 later
    "lr": 0.01,  #Change this back to 0.01 later
    "seed": 0,
    "fold_idx": 0,
    "num_layers": 5,
    "num_mlp_layers": 2,
    "hidden_dim": 64,
    "final_dropout": 0.5,
    "graph_pooling_type": 'sum',
    "neighbor_pooling_type": 'sum',
    "learn_eps": 'store_true',
    'degree_as_tag': 'store_true',
    'filename': 'output.txt'
    
})





loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
labels = tf.constant([graph.label for graph in train_graphs])

model = GraphCNN(args.num_layers, args.num_mlp_layers, args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type)

optimizer = tf.keras.optimizers.Adam(lr = args.lr)



#def train(loss,model,opt,original):    
def train(args,model,train_graphs,opt,epoch):
  total_iters = args.iters_per_epoch
  pbar = tqdm(range(total_iters),unit = 'batch')
  
  loss_accum = 0
  for pos in pbar:
    selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
    batch_graph = [train_graphs[idx] for idx in selected_idx]
    labels = tf.constant([graph.label for graph in batch_graph])
    loss_accum = 0
    with tf.GradientTape() as tape:
      output = model(batch_graph)

      loss = loss_object(labels,output)

      
    gradients = tape.gradient(loss,model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)
    loss_accum += loss
    
    #report
    pbar.set_description(f'epoch: {epoch}')
    
    #return reconstruction_error
  average_loss = loss_accum/total_iters
  print(f'loss training: {average_loss}')
  return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        #output.append(model([graphs[j] for j in sampled_idx]).detach())        
        output.append(model([graphs[j] for j in sampled_idx]))
    return tf.concat(output,0)

  
def tf_check_acc(pred,labels):
    pred = tf.cast(pred,tf.int32)
    correct = tf.equal(pred,labels)
    answer = 0
    for element in correct:
      if element:
        answer +=1
    return answer
  
def test(args, model, train_graphs, test_graphs, epoch):
    output = pass_data_iteratively(model, train_graphs)
    #print(f'This is the output: {output}')
    #pred = output.max(1, keepdim=True)[1]  #This gives the index of the output with the largest number
    pred = tf.argmax(output,1)
    #labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    labels = tf.constant([graph.label for graph in train_graphs])
    #print(f'These are the labels: {labels}\n\nThese are the predictions: {pred}')
    
    #correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    #print(pred,labels)
    correct = tf_check_acc(pred,labels)
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    #pred = output.max(1, keepdim=True)[1]
    pred = tf.argmax(output,1)
    labels = tf.constant([graph.label for graph in test_graphs])
    #labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    #correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    correct = tf_check_acc(pred,labels)
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test



for epoch in range(1, args.epochs + 1):
    #global_step = global_step + 1
    if epoch % 50 == 0:
      optimizer.lr = optimizer.lr * 0.5    
    print (optimizer.lr)
    avg_loss = train(args, model, train_graphs, optimizer, epoch)
    acc_train, acc_test = test(args, model, train_graphs, test_graphs, epoch)

    if epoch % 50 == 0:
      optimizer.lr = optimizer.lr * 0.5
    
    if not args.filename == "":
        with open(args.filename, 'w') as f:
            f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
            f.write("\n")
    print("")

    print(model.eps)
