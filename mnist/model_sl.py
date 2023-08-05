import pickle
def Saver(x,y,b):
    for i in x:
        with open("model"+i+".txt","wb") as f:
            pickle.dump(y.params[i], f)
def dLoader(model):
    for i in range(model.layer+1):
        with open("model"+'w'+str(i+1)+".txt","rb") as fw:
            with open("model"+'b'+str(i+1)+".txt","rb") as fb:
                model.layers['Linear'+str(i+1)].w = pickle.load(fw)
                model.layers['Linear'+str(i+1)].b = pickle.load(fb)

