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
from processing_ColumnKeep import DataPreprocessing_columnKeep
#from Bulid_2layers_model import seq2seq
from nobatchorm_and_bi import seq2seq
from sklearn.metrics import classification_report,recall_score,precision_score,f1_score
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.set_option('display.width', 1000)
#啟動hualos:python hualos/api.py

def train_test_split(label,features,percent,embed1,embed2,column) :
    msk = np.random.rand(len(features))<percent
    l_onehot = pd.get_dummies(data=label)
    l = l_onehot.values
    y_train,y_test = l[msk],l[~msk] 
    scalar = MinMaxScaler(feature_range=(0,1))
    x = features.values
    if column =="no":
        x = np.nan_to_num(x)
        x_scale = scalar.fit_transform(X = x)
        x_train,x_test = x_scale[msk],x_scale[~msk]
    else:
        x_concate = np.concatenate([x,embed1,embed2],axis = 1)
        x_concate = np.nan_to_num(x_concate)
        x_scale = scalar.fit_transform(X = x_concate)#
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
total_start = time.time()
def main():
    num = 8
    binary = True
    train_num = 2000000
    epochs = 8
    prediction_list = []
    recall_list = []
    f1_score_list = []
    acc_list = []
    pd.options.display.max_columns = None
    column = input("是否使用已丟棄多餘欄位之資料? yes/no : ")
    if column == "yes":
        ISCX = DataPreprocessing_columnKeep(foldpath="/home/jim/MalwareData/ISCX_columndrop/",num = num,label=True,binary=binary)
        print("合併了%d個表格"%ISCX.num)
        rawDF = ISCX.ReadIntoDF()
        sortDF = ISCX.move_label_remove_zero_feature(rawDF)
        label,features,embed_srcport,embed_dstport = ISCX.get_label_feature(sortDF,train_num)#超過500就捨棄
        print("\n有{}個Label".format(len(label)))
        print("特徵數目為：{}\n".format(features.shape[1]))
    if column =="no":
        ISCX_Dr = DataPreprocessing(foldpath="/home/jim/MalwareData/ISCX/",num = num,label=True,binary=binary)
        print("合併了%d個表格"%ISCX_Dr.num)
        rawDF = ISCX_Dr.ReadIntoDF_Dr()
        sortDF = ISCX_Dr.move_label_remove_zero_feature_Dr(rawDF)
        label,features = ISCX_Dr.get_label_feature_Dr(sortDF,train_num)#超過500就捨棄
        print("\n有{}個Label".format(len(label)))
        print("特徵數目為：{}\n".format(features.shape[1]))
        embed_srcport=0
        embed_dstport=0

    n_features = features.shape[1]
    
    for times in range(30):
        print("第",times+1,"次\n")
        if column=="yes":
            x_train,y_train,x_test,y_test,L_onehot = train_test_split(label = label,features = features,percent = 0.9,embed1 = embed_srcport,embed2 = embed_dstport,column=column)
        else:
            x_train,y_train,x_test,y_test,L_onehot = train_test_split(label = label,features = features,percent = 0.9,embed1 = embed_srcport,embed2 = embed_dstport,column=column)
        print("L_onehot shape:",L_onehot.shape)
        n_classes = y_test.shape[1]
        n_units = 32
        datalist = [x_train,y_train,x_test,y_test]
        namelist = ["x_train","y_train","x_test","y_test"]
        dataDict = {}
    

        #變成三維，存成字典
        for i in range(len(datalist)):
            datalist[i] = datalist[i].reshape(len(datalist[i]),1,-1)
            datalist[i] = np.nan_to_num(datalist[i])
            dataDict[namelist[i]] = datalist[i]
        print(x_train.shape)
        print(dataDict["x_train"].shape)
        
        delay_X_train = generate_delayData(dataDict["x_train"],n_features=n_features)
        delay_x_test = generate_delayData(dataDict["x_test"],n_features)
        mapdict = dict_generater(L_onehot)

        #建構模型
        model_constructor = seq2seq(dataDict["x_train"],dataDict["y_train"],dataDict["x_test"],dataDict["y_test"],n_units,binary = binary)
        total_model,encoder,decoder = model_constructor.model_build()

        #訓練模型
        t_start = time.time()
        history = model_constructor.train(delaydata = delay_X_train,epoch=epochs)
        t_stop = time.time()
        print("\n total cost %.3f minutes to train \n"%((t_stop-t_start)/60))
        
        #進行預測
        print("Predicting ....")
        prediction,labels,score ,raw_p,raw_l,acc= model_constructor.predict_sequence(mapdict,delay_x_test)
        scorelist = score.tolist()
        cross = pd.crosstab(labels,prediction,rownames=["Label"],colnames=["prediction"])
        print(cross)
        print("\n",classification_report(labels,prediction))
        print("\n","Precision :",precision_score(raw_l,raw_p))#2分類不用average
        print("\n","Recall :",recall_score(raw_l,raw_p,))#average="weighted"))
        print("\n","F1-Score : ",f1_score(raw_l,raw_p,))#average="weighted"))
        total_end = time.time()
        print("總共花了{}分鐘跑完".format((total_end-total_start)/60))
        print("total epochs : %d"%(epochs))
        #print(show_train_history(history,"acc","loss"))
        #print(model_constructor.AUC_plot(scorelist,"CICIDS2017"))
        if column == "yes":
            print("此次實驗為已丟棄冗餘特徵且使用嵌入法之結果")
        else:
            print("此次實驗為已丟棄冗餘特徵但沒有嵌入之結果")
        prediction_list.append(precision_score(raw_l,raw_p,))#average="weighted"))
        recall_list.append(recall_score(raw_l,raw_p,))#average="weighted"))
        f1_score_list.append(f1_score(raw_l,raw_p,))#average="weighted"))
        acc_list.append(acc)
        calculate = pd.DataFrame({"Accuracy":acc_list,"Prediction":prediction_list,"Recall":recall_list,"f1_score":f1_score_list})
        calculate.to_csv("/home/jim/Desktop/實驗紀錄/30times_calculate_multiclass_withRF_nobatchnorm_%d.csv"%times)
        #fpr,tpr,auc = model_constructor.AUC_plot(scorelist,"CICIDS2017")
        # rocDF = pd.DataFrame({"fpr":fpr,"tpr":tpr,"auc_value":auc})
        # rocDF.to_csv("/home/jim/Desktop/實驗紀錄/auc_value_CICIDS2017_%d.csv"%i)


if __name__ =='__main__':
    main()
# print("Saving model ....")
# total_model.save("/home/jim/pywork/Models/200000_2layers_total_model.h5")
# encoder.save("/home/jim/pywork/Models/200000_2layers_encoder.h5")
# decoder.save("/home/jim/pywork/Models/200000_2layers_decoder.h5")
