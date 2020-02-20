import time
#import numba as nb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed
import numpy as np
from keras.models import Model
from keras.layers import Input
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from processing_ColumnDrop import DataPreprocessing
from Bulid_2layers_model import seq2seq
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from AttentionSeqtoSeqModel import AttentionSeqtoSeq
from seq2seq import AttentionSeq2Seq
pd.options.display.max_columns = None

def train_test_split(label,features,percent) :
    msk = np.random.rand(len(features))<percent
    l_onehot = pd.get_dummies(data=label)
    l = l_onehot.values
    y_train,y_test = l[msk],l[~msk] 
    scalar = MinMaxScaler(feature_range=(0,1))
    x = features.values
    x_scale = scalar.fit_transform(X = x)#
    x_train,x_test = x_scale[msk],x_scale[~msk]
    return x_train,y_train,x_test,y_test,l_onehot
def generate_delayData(data,n_features):
    # zeros = np.zeros(n_features).reshape(1,1,n_features)
    # return np.insert(data[:-1],0,[zeros],0) #此為原來的
    zero = np.zeros(1).reshape(1,1,1)
    newdata = np.insert(data,0,zero,axis=2)[:,:,:-1]
    return newdata
def dict_generater(L_onehot):
    mapdict = {}
    for i in range(len(list(L_onehot))):
        mapdict[i] = list(L_onehot)[i]
    return mapdict
def show_train_history(train_history,t,l):
    plt.plot(train_history.history[t])
    plt.plot(train_history.history[l])
    plt.title('Train History')
    plt.ylabel(t)
    plt.xlabel("Epoch")
    plt.legend(['train',"loss"],loc="lower right")
    plt.show()

def predict_sequence(model,mapdict,xtest,ytest):
    pre = model.predict(xtest)
    prediction = []
    label = []
    correct =0
    total = len(xtest)
    for i in pre:
        p = np.argmax(i)
        prediction.append(mapdict[p])
    
    for t in range(len(ytest)):
        if mapdict[np.argmax(ytest[t])]==prediction[t]:
            correct+=1
        label.append(mapdict[np.argmax(ytest[t])])
    prediction = np.array(prediction)
    label = np.array(label)
    print("accuracy is :{}".format(correct/total))
    print(pd.crosstab(label,prediction,rownames=["Label"],colnames=["prediction"]))
    print("\n",classification_report(label,prediction))
total_start = time.time()
#@nb.jit(nopython=True)
pd.options.display.max_columns = None
ISCX = DataPreprocessing(foldpath="/home/jim/MalwareData/ISCX/",num = 8,label=True,binary=True)
rawDF = ISCX.ReadIntoDF()
sortDF = ISCX.move_label_remove_zero_feature(rawDF)
label,features = ISCX.get_label_feature(sortDF,2000000)#超過500就捨棄
print("\n有{}個Label".format(len(label)))
print("特徵數目為：{}\n".format(features.shape[1]))

n_features = features.shape[1]
x_train,y_train,x_test,y_test,L_onehot = train_test_split(label = label,features = features,percent = 0.9)
n_classes = y_test.shape[1]
n_units = 32
datalist = [x_train,y_train,x_test,y_test]
namelist = ["x_train","y_train","x_test","y_test"]
dataDict = {}
epochs = 10
depth = 4
#變成三維，存成字典
for i in range(len(datalist)):
    datalist[i] = datalist[i].reshape(len(datalist[i]),1,-1)
    dataDict[namelist[i]] = datalist[i]
print(x_train.shape)
print(dataDict["x_train"].shape)

delay_X_train = generate_delayData(dataDict["x_train"],n_features=n_features)
delay_x_test = generate_delayData(dataDict["x_test"],n_features)
mapdict = dict_generater(L_onehot)

#Attention SeqtoSeq with 2 layers Dense
m1 = AttentionSeqtoSeq(depth=depth,input_dim=n_features,output_dim=n_classes,input_length=1,output_length=1)
m1.compile(loss="mse",optimizer="adam",metrics=["acc"])
m1.fit(dataDict["x_train"],dataDict["y_train"],epochs=epochs,)
predict_sequence(m1,mapdict,dataDict["x_test"],dataDict["y_test"])

#Attention SeqtoSeq
m2 = AttentionSeqtoSeq(depth=depth,input_dim=n_features,output_dim=n_classes,input_length=1,output_length=1)
m2.compile(loss="mse",optimizer="adam",metrics=["acc"])
m2.fit(dataDict["x_train"],dataDict["y_train"],epochs=epochs,)
predict_sequence(m2,mapdict,dataDict["x_test"],dataDict["y_test"])

#MLP
m3 = Sequential()
m3.add(Dense(100,input_shape = (None,n_features)))
m3.add(Dense(50))
m3.add(Dense(25))
m3.add(Dense(n_classes,activation="softmax"))
m3.compile(loss = "categorical_crossentropy",optimizer="adam",metrics = ["acc"])
#x_reshape = dataDict["x_train"].reshape(dataDict["x_train"].shape[0],dataDict["x_train"].shape[2])
m3.fit(x =dataDict["x_train"],y = dataDict["y_train"],epochs = epochs)
predict_sequence(m3,mapdict,dataDict["x_test"],dataDict["y_test"])

total_stop =time.time()
print("總共花了{}分鐘跑完".format((total_start-total_stop)/60))