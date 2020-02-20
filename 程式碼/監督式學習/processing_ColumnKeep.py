import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ProcessBar import ShowProcess
from keras.layers import Embedding
from keras.models import Sequential
from keras import regularizers
#pd.options.display.max_columns = None

class DataPreprocessing_columnKeep:
    def __init__(self,foldpath,num,label,binary = False):
        self.foldpath = foldpath
        self.num = num
        self.datalist = os.listdir(self.foldpath)
        self.label = label
        self.binary = binary
        print('資料夾{}內，共有{}個檔案'.format(self.foldpath,len(self.datalist)))
    
    def ReadIntoDF(self):
        totalDF = pd.DataFrame([])
        processbar = ShowProcess(self.num,infoDone="\n資料表格合併完成 !")
        for i in range(self.num):
            processbar.show_process()
            oneDF = pd.read_csv(self.foldpath+self.datalist[i],encoding='latin1',engine='python')
            totalDF = totalDF.append([oneDF])
        self.totalDF = totalDF
        return totalDF
    def show_class_percentage(self):
        sns.set(font_scale=5)
        f,ax = plt.subplots(figsize=(60,20))
        ax.plot()
        sns.countplot(self.totalDF[' Label'],ax=ax)
    def move_label_remove_zero_feature(self,data):
        
        if self.label==True:
            list1 = list(data)
            list1.insert(0,list1.pop(list1.index(' Label')))
        else :
            list1 = ['ACK Flag Cnt','Active Max',"Active Min","Active Std","Pkt Size Avg",'Bwd Seg Size Avg','Fwd Seg Size Avg',
           "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Header Len","Bwd IAT Max","Bwd IAT Mean",'Bwd IAT Min',"Bwd IAT Std",
           "Bwd PSH Flags","Bwd Pkt Len Mean","Bwd Pkt Len Min","Bwd Pkt Len Std","Bwd Pkts/s","Bwd URG Flags",
           "CWE Flag Count","Dst IP","Dst Port","Down/Up Ratio","ECE Flag Cnt","Flow Duration","Flow IAT Max",
           "Flow IAT Mean","Flow IAT Min","Flow IAT Std","Flow Pkts/s","Fwd Blk Rate Avg","Fwd Pkts/b Avg","Fwd Header Len",
           "Fwd IAT Max","Fwd IAT Mean","Fwd IAT Min","Fwd IAT Std","Fwd Pkt Len Max","Fwd Pkt Len Mean","Fwd Pkt Len Min",
           "Fwd Pkt Len Std","Fwd URG Flags","Idle Max","Idle Min","Idle Std","Init Bwd Win Byts","Label","Pkt Len Max",
           "Pkt Len Min","PSH Flag Cnt","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Protocol","RST Flag Cnt","SYN Flag Cnt",
           "Src IP","Src Port","Subflow Bwd Byts","Subflow Bwd Pkts","Subflow Fwd Byts","Timestamp","Tot Bwd Pkts","Tot Fwd Pkts",
           "TotLen Bwd Pkts","URG Flag Cnt","Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Bwd Blk Rate Avg","Bwd IAT Tot",
           "Bwd Pkt Len Max","FIN Flag Cnt","Flow Byts/s","Flow ID","Fwd Byts/b Avg","Fwd IAT Tot","Fwd PSH Flags",
           "Fwd Pkts/s","Idle Mean","Init Fwd Win Byts","Subflow Fwd Pkts","TotLen Fwd Pkts"]
            list1.insert(0,list1.pop(list1.index('Label')))
        data2 = data.loc[:, list1]
        print("換位過後的資料型態為：{}".format(data2.shape))
        
        if self.label==True:
            sort_DF = data2.sort_values([' Timestamp'],ascending=True)
        else:
            sort_DF = data2.sort_values(['Timestamp'],ascending=True)
        sort_DF.drop_duplicates(keep='first',inplace=True)
        print("排序，刪除重複欄位過後的資料型態為：{}".format(sort_DF.shape))
        # keylist = list(sort_DF)
        # for i in keylist:
        #     if list(sort_DF[i]).count(0)==len(sort_DF):
        #         print(i+" 的特徵數值全為 0，故丟棄")
        #         sort_DF.drop([i,],axis=1,inplace=True)
        # print("刪除都為0的欄位，資料型態為:{}".format(sort_DF.shape))
        return sort_DF

    def get_label_feature(self,data,slice_number):
        
        def augFeatures(data,time):
            data[time] = pd.to_datetime(data[time],dayfirst=True,errors="coerce")
            data["month"] = data[time].dt.month
            data["date"] = data[time].dt.day
            data["hour"] = data[time].dt.hour
            data["minutes"] = data[time].dt.minute
            return data
        
        def Embedding_generater(data,column_name,output_dim):
            print("正在 Embedding 特徵： "+column_name+" 轉成",output_dim,"維")
            model = Sequential()
            max_dim = len(list(data.groupby(column_name).groups))
            model.add(Embedding(input_dim=max_dim,output_dim=output_dim,input_length=1,embeddings_regularizer = regularizers.l1(0.01)))
            embed_col = model.predict(data[column_name])
            embed_col = embed_col.reshape(embed_col.shape[0],-1)
            return embed_col

        binaryDict = {}
        data.reset_index(drop=True,inplace=True)
        if self.label==True:
            containDF = pd.DataFrame([])
            class_sortDF = data.groupby(" Label")#為了不要讓某個種類太多，用label當
            print("資料類別及數目總覽：\n",class_sortDF.size())  
            for key in list(class_sortDF.groups):
                print("處理類別{}".format(key))
                if len(class_sortDF.get_group(key))<slice_number:
                    a = pd.DataFrame(class_sortDF.get_group(key))
                else :
                    a = pd.DataFrame(class_sortDF.get_group(key)[:slice_number])#超過slice_number的話取slice_number筆
        
                #轉換為二分類
                if self.binary == True:
                    print("轉為2分類")
                    if key !='BENIGN':
                        binaryDict[key] = "Malware"
                    else :pass
                else:pass
                containDF = containDF.append([a])
            containDF = augFeatures(containDF," Timestamp")
            l = containDF[" Label"].copy()
            if self.binary == True:
                l.replace(binaryDict,inplace = True)
            else:pass
            
            print("\nEmbedding ... ")
            embed_srcport = Embedding_generater(containDF," Source Port",2)
            embed_dstport = Embedding_generater(containDF," Destination Port",2)

            #containDF = pd.get_dummies(data =containDF,columns=[" Protocol"] )會使誤判率提高
            print("丟棄多餘特徵")
            containDF.drop(columns=[" Label"," Timestamp"," Destination IP"," Destination Port"," Source IP",
                   " Source Port","Flow ID","Flow Bytes/s",' Fwd Header Length.1',"External IP"," Protocol",#此行以後為經過ＲＦ判定為不重要的特徵，Eternal IP 為部份資料才有的特徵,' Fwd Header Length.1'是有label資料才有的
                    ],inplace=True)#Destination Port,source port 要做Embedding (protocol經實驗過不適合做獨熱編碼)
        else:
            containDF = data.copy()
            containDF = augFeatures(containDF,"Timestamp")
            containDF.drop(columns=["Label","Timestamp","Dst IP","Dst Port","Src IP",
                   "Src Port","Flow ID","Flow Byts/s","Protocol",
                   "Bwd Byts/b Avg","Fwd URG Flags","Bwd Blk Rate Avg","Bwd Pkts/b Avg","Bwd URG Flags","Fwd Blk Rate Avg",
                   "Fwd Pkts/b Avg","Bwd PSH Flags","Flow Pkts/s","ECE Flag Cnt","CWE Flag Count","RST Flag Cnt","Fwd Byts/b Avg"],inplace=True)#Destination Port,source port,protocol 要做onehot
        
        with pd.option_context('mode.use_inf_as_null', True):#改過
            containDF = containDF.fillna(value=0.0)
        containDF=containDF.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))#最大最小標準化
        #print(list(containDF))
        containDF[list(containDF)] = containDF[list(containDF)].astype("float32")
        if self.label==True:
            return l,containDF,embed_srcport,embed_dstport
        
        else :
            features=containDF.values
            print("特徵的形狀為：{}".format(features.shape))
            return features
