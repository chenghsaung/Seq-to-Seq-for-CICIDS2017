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
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import classification_report,recall_score,precision_score,f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.models import Sequential
from keras import regularizers

def generate_delayData(data):
    zero = np.zeros(1).reshape(1,1,1)
    newdata = np.insert(data,0,zero,axis=2)[:,:,:-1]
    return newdata

def dict_generater(L_onehot):
    mapdict = {}
    for i in range(len(list(L_onehot))):
        mapdict[i] = list(L_onehot)[i]
    return mapdict

def Embedding_generater(data,column_name,output_dim):
    print("正在 Embedding 特徵： "+str(column_name)+" 轉成",output_dim,"維")
    new_2 = []
    for t in data[column_name]:
        newt = t.decode("ascii")
        new_2.append(newt)
    new2df = pd.DataFrame(new_2)
    token = Tokenizer(num_words = len(list(new2df.groupby(0).groups))+1)
    token.fit_on_texts(new_2)
    new2_sequence = token.texts_to_sequences(new_2)
    new_2_pad = sequence.pad_sequences(new2_sequence,maxlen=1)
    model = Sequential()
    max_dim = len(list(data.groupby(column_name).groups))
    model.add(Embedding(input_dim=max_dim,output_dim=output_dim,input_length=1,embeddings_regularizer = regularizers.l1(0.01)))
    embed_col = model.predict(new_2_pad)
    embed_col = embed_col.reshape(embed_col.shape[0],-1)
    return embed_col

raw_data_dict = fetch_kddcup99()
kdd99 = pd.DataFrame(raw_data_dict["data"])
print("KDDCup99共有{}筆流量紀錄,{}個特徵".format(kdd99.shape[0],kdd99.shape[1]))
kdd_onehot = pd.get_dummies(data=kdd99,columns=[1,3])

#做隨機森林後不重要的特徵
kdd_onehot.drop(columns=[20,19,"3_b'S3'","3_b'RSTOS0'",14],inplace=True)

binDict = {}
labels = pd.DataFrame(raw_data_dict["target"])
for l in list(labels.groupby([0]).groups):
    if l  == b'normal.':
        binDict[l]="BENIGN"
    else:
        binDict[l] = "Malware"

embed_col2 = Embedding_generater(kdd_onehot,2,2)
embed_1 = pd.Series(embed_col2[:,0],name="embed_1")
embed_2 = pd.Series(embed_col2[:,1],name="embed_2")
kdd_onehot.drop(columns=[2],inplace=True)
totalDF = pd.concat([kdd_onehot,embed_1,embed_2],axis = 1)

#標籤二元化
labels.replace(binDict,inplace = True)

#標籤 one_hot
labels_onehot = pd.get_dummies(labels)

#將nan轉為0
total_features = np.nan_to_num(totalDF.values)

#最大最小標準化
scalar = MinMaxScaler(feature_range=(0,1))
raw_features = scalar.fit_transform(total_features)

raw_labels = labels_onehot.values

raw_features = raw_features.reshape(raw_features.shape[0],1,-1)
raw_labels = raw_labels.reshape(raw_labels.shape[0],1,-1)
labels_onehot.rename(columns={"0_BENIGN":"BENIGN",'0_Malware':'Malware'},inplace=True)

acclist = []
precisionlist = []
recalllist = []
f1_scorelist = []

for i in range(1):
    print("第",i+1,"次實驗\n")
    #訓練 測試分開
    msk = np.random.rand(len(raw_features))<0.8
    train_features = raw_features[msk]
    train_labels = raw_labels[msk]
    test_features = raw_features[~msk]
    test_labels = raw_labels[~msk]

    #產生delay資料
    train_features_delay = generate_delayData(train_features)
    test_features_dalay = generate_delayData(test_features)

    print("Model Building ....")
    model_constructor = seq2seq(train_features,train_labels,test_features,test_labels,32,binary = True)
    total_model,encoder,decoder = model_constructor.model_build()
    mapdict = dict_generater(labels_onehot)

    history = model_constructor.train(delaydata = train_features_delay,epoch=12)

    print("Predicting ....")
    prediction,labels,score,raw_p,raw_l,acc = model_constructor.predict_sequence(mapdict,delay_x_test=test_features_dalay)

    #將 Malware -> 0, BENIGN -> 1
    mapdict_reverse = {"BENIGN":0,"Malware":1}
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
    record.to_csv("/home/jim/Desktop/實驗紀錄/KDDCup99實驗紀錄/30times_%d.csv"%i)
    auc = metrics.roc_auc_score(replace_labels,scorelist)
    rocDF = pd.DataFrame({"fpr":fpr,"tpr":tpr,"auc_value":auc})
    rocDF.to_csv("/home/jim/Desktop/實驗紀錄/KDDCup99實驗紀錄/auc_value_%d.csv"%i)

    # plt.plot(fpr,tpr,label="data KDDCup99, auc=%.8f"%auc)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.ylim(0,1.1)
    # plt.legend(loc = "upper right")
    # plt.savefig("/home/jim/Desktop/實驗紀錄/KDDCup99實驗紀錄/aucplot_%d"%i)
    # #plt.show()
    # plt.close()