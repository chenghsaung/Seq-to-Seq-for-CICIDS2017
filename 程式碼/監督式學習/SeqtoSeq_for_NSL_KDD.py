import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import Input
from Bulid_2layers_model import seq2seq
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import classification_report,recall_score,precision_score,f1_score
from keras.layers import Embedding
from keras.models import Sequential
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


pd.options.display.max_columns = None

def turn_to_bin_label(l):
    binDict = {}
    for la in list(l):
        if la == 'normal':
            #print(la,"->","normal")
            binDict[la] = "BENIGN"
        else:
            #print(la,"->","Malware")
            binDict[la] = "Malware"
    print(binDict)
    l.replace(binDict,inplace = True)
    return l

def generate_delayData(data):
    zero = np.zeros(1).reshape(1,1,1)
    newdata = np.insert(data,0,zero,axis=2)[:,:,:-1]
    return newdata

def Embedding_generater(data,column_name,output_dim):
    print("正在 Embedding 特徵： "+str(column_name)+" 轉成",output_dim,"維")
    max_dim = len(list(data.groupby(column_name).groups))
    token = Tokenizer(num_words = max_dim)
    feature_list = list(data[column_name])
    token.fit_on_texts(feature_list)
    feature_sequence = token.texts_to_sequences(feature_list)
    feature_pad = sequence.pad_sequences(feature_sequence,maxlen=1)
    model = Sequential()
    model.add(Embedding(input_dim=max_dim,output_dim=output_dim,input_length=1,embeddings_regularizer = regularizers.l1(0.01)))
    embed_col = model.predict(feature_pad)
    embed_col = embed_col.reshape(embed_col.shape[0],-1)
    return embed_col

def main():
    KDDTrain_plus = pd.read_csv("/home/jim/Desktop/malware sample/KDD/NSL_KDD-master/KDDTrain+.csv")
    #total_NSLkdd = pd.concat([KDDTrain_plus,twentypercent],axis = 0)
    total_NSLkdd = KDDTrain_plus
    KDDTest_plus = pd.read_csv("/home/jim/Desktop/malware sample/KDD/NSL_KDD-master/KDDTest+.csv")
    print("total_NSLkdd shape :",total_NSLkdd.shape)
    print("KDDTest_plus shape :",KDDTest_plus.shape)

    l_train_plus = total_NSLkdd["normal"].copy() # total_NSLkdd = KDDTrain_plus
    l_test_plus = KDDTest_plus["neptune"].copy()

    #標記型特徵onehot
    total_NSLkdd_onehot = pd.get_dummies(data=total_NSLkdd,columns=["tcp","SF","0.2","0.7","0.16","0.17"])
    KDDTest_plus_onehot = pd.get_dummies(data=KDDTest_plus,columns=["tcp","REJ","0.3","0.8","0.17","0.18"])
    total_NSLkdd_onehot.drop(columns = ["normal"],inplace=True)
    KDDTest_plus_oneho
    ["neptune"],inplace = True)

    #把Label二元化
    train_bin_label = turn_to_bin_label(l_train_plus)
    test_bin_label = turn_to_bin_label(l_test_plus)

    #對二元化的標籤做讀熱編碼
    train_label_onehot = pd.get_dummies(train_bin_label)
    test_label_onehot = pd.get_dummies(test_bin_label)

    #嵌入高維度特徵，把service刪除，並把嵌入的特徵結合進訓練以及測試資料集
    embed_private = Embedding_generater(KDDTest_plus_onehot,"private",2)
    embed_ftp_data = Embedding_generater(total_NSLkdd_onehot,"ftp_data",2)
    total_NSLkdd_onehot.drop(columns=["ftp_data"],inplace = True)
    KDDTest_plus_onehot.drop(columns=["private"],inplace = True)
    total_NSLkdd_onehot_embed = np.concatenate([total_NSLkdd_onehot.values,embed_ftp_data],axis=1)
    KDDTest_plus_onehot_embed = np.concatenate([KDDTest_plus_onehot.values,embed_private],axis=1)



    #把特徵轉為矩陣，nan轉為0
    train_features = np.nan_to_num(total_NSLkdd_onehot_embed)
    test_features = np.nan_to_num(KDDTest_plus_onehot_embed)

    #最大最小標準化
    scalar = MinMaxScaler(feature_range=(0,1))
    train_features = scalar.fit_transform(X = train_features)
    test_features = scalar.fit_transform(X = test_features)

    #Label turn to matrix
    train_labels = train_label_onehot.values
    test_labels = test_label_onehot.values

    total_features = train_features
    total_labels = train_labels

    # total_features = np.concatenate([train_features,test_features],axis = 0)
    # total_labels = np.concatenate([train_labels,test_labels],axis = 0)

    #chaange features, labels to 3 dim
    total_features = total_features.reshape(total_features.shape[0],1,-1)
    test_features = test_features.reshape(test_features.shape[0],1,-1) #此為測試時用的資料，而不是隨機取
    total_labels = total_labels.reshape(total_labels.shape[0],1,-1)
    test_labels = test_labels.reshape(test_labels.shape[0],1,-1) #此為測試時用的資料，而不是隨機取

    
    #產生延遲資料
    total_features_delay = generate_delayData(total_features)
    test_features_dalay = generate_delayData(test_features)
    print("test_features_dalay shape : ",test_features_dalay.shape)
    acclist = []
    precisionlist = []
    recalllist = []
    f1_scorelist = []
    for i in range(1):
        print("第",i+1,"次實驗\n")
        # Build th model 
        model_constructor = seq2seq(total_features,total_labels,test_features,test_labels,32,binary = True)
        total_model,encoder,decoder = model_constructor.model_build()
        # train model
        history = model_constructor.train(delaydata = total_features_delay,epoch=20)

        mapdict = {0:"Malware",1:"BENIGN"}
        prediction,labels,score,raw_p,raw_l,acc = model_constructor.predict_sequence(mapdict,delay_x_test=test_features_dalay)#產生預測標籤,實際標籤,預測值score

        #將 Malware -> 0, BENIGN -> 1
        mapdict_reverse = {"Malware":0,"BENIGN":1}
        replace_labels = pd.DataFrame(labels).replace(mapdict_reverse)
        #把score從矩陣轉成清單
        replace_labels = list(replace_labels[0])
        scorelist = score.tolist()

        fpr, tpr, thresholds = metrics.roc_curve(replace_labels, scorelist,drop_intermediate=True)

        print(pd.crosstab(labels,prediction,rownames=["Label"],colnames=["prediction"]))
        print("\n",classification_report(labels,prediction))
        print("\n","Precision :",precision_score(raw_l,raw_p))#2分類不用average
        print("\n","Recall :",recall_score(raw_l,raw_p,))#average="weighted"))
        print("\n","F1-Score : ",f1_score(raw_l,raw_p,))#average="weighted"))
        
        precisionlist.append(precision_score(raw_l,raw_p,))#average="weighted"))
        recalllist.append(recall_score(raw_l,raw_p,))#average="weighted"))
        f1_scorelist.append(f1_score(raw_l,raw_p,))#average="weighted"))
        acclist.append(acc)

        record = pd.DataFrame({"Accuracy":acclist,"Precision":precisionlist,"Recall":recalllist,"f1_score":f1_scorelist})
        #record.to_csv("/home/jim/Desktop/實驗紀錄/NSL_KDD實驗紀錄2/30times_calculate_%d.csv"%i)
        auc = metrics.roc_auc_score(replace_labels,scorelist)
        rocDF = pd.DataFrame({"fpr":fpr,"tpr":tpr,"auc_value":auc})
        rocDF.to_csv("/home/jim/Desktop/實驗紀錄/NSL_KDD實驗紀錄2/auc_value_NSLKDD_%d.csv"%i)
        
        # plt.plot(fpr,tpr,label="data NSL_KDD, auc=%.8f"%auc)
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.ylim(0,1.1)
        # plt.legend(loc = "upper right")
        # plt.savefig("/home/jim/Desktop/實驗紀錄/NSL_KDD實驗紀錄2/aucplot_%d"%i)
        # plt.close()

if __name__=="__main__":
    main()
