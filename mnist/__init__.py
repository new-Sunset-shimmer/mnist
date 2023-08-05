import mlp
import data
import model_sl
def run(layers,max_epoch = 100,batch_size = 500):
    network = mlp.MLP(layers)
    train_x, test_x, train_y, test_y = data.Dataset().data()
    learning_rate = 0.5
    wb_arrange = ['w1', 'b1']
    test_stop=0
    pat = 10
    for i in range(network.layer):
        wb_arrange.append('w'+str(i+2))
        wb_arrange.append('b'+str(i+2))
        
    for epoch in range(max_epoch):
        for i in range(train_x.shape[0] // batch_size):
            start_index = i * batch_size
            end_index = start_index + batch_size
            x_batch = train_x[start_index:end_index]
            t_batch = train_y[start_index:end_index]
            grad = network.gradient(x_batch, t_batch)
            for key in wb_arrange:
                network.params[key] -= learning_rate * grad[key]

            # network.loss(x_batch, t_batch)
        train = network.accuracy(train_x, train_y)
        test = network.accuracy(test_x, test_y) 
        train_loss = network.loss(train_x, train_y)
        test_loss = network.loss(test_x, test_y)
        print('Epoch  {:d}'.format(epoch))
        print('[Train]  Loss : {:f}    Acc :  {:.4f}'.format(train_loss, train*100))
        print('[Test]  Loss : {:f}    Acc :  {:.4f}'.format(test_loss, test*100))
        # if(abs(test*100-test_stop)<0.2):
        #     pat +=1
        #     if pat >3:
        #         break
        # else:
        #     pat = 0
        # test_stop = test*100
        if(test*100>test_stop):
            test_stop = test*100
            pat = 0
        elif(pat<1):
            pat +=1
        else:
            break
    model_sl.Saver(wb_arrange,network,layers)
def test():
    layer = layer_open()
    model = mlp.MLP(layer)
    import pickle
    for i in range(model.layer+1):
        with open("model"+'w'+str(i+1)+".txt","rb") as fw:
            with open("model"+'b'+str(i+1)+".txt","rb") as fb:
                model.layers['Linear'+str(i+1)].w = pickle.load(fw)
                model.layers['Linear'+str(i+1)].b = pickle.load(fb)
    train_x, test_x, train_y, test_y = data.Dataset().data()
    acc = model.accuracy(test_x, test_y)
    loss = model.loss(test_x, test_y)
    print('Loss : {:f}    Acc :  {:.4f}'.format(loss, acc*100))
def layer_save(layer):
     with open("model_layers.txt", 'w') as f:
        for item in layer:
            f.write(str(item) + '\n')
def layer_open():
    layer = []
    with open("model_layers.txt", 'r') as f:
        for line in f:
            item = line.strip()
            layer.append(int(item))
    return layer
if __name__ == '__main__':
    print("epoch:")
    epoch = int(input())
    print("batch_size:")
    batch_size = int(input())
    print("hidden_layer_size:")
    layer_size = int(input())+2
    layers = [784]
    print("layer:")
    for _ in range(layer_size-2):
        layers.append(int(input()))
    layers.append(10)
    print("run:")
    run(layers,epoch,batch_size)
    layer_save(layers)
    test()