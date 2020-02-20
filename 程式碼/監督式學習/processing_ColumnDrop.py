import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#from sklearn.
#pd.options.display.max_columns = None

class DataPreprocessing:
    print("name =",__name__)
    def __init__(self,foldpath,num,label,binary = False):
        self.foldpath = foldpath
        self.num = num
        self.datalist = os.listdir(self.foldpath)
        self.label = label
        self.binary = binary
        print('資料夾{}內，共有{}個檔案'.format(self.foldpath,len(self.datalist)))
    
    def ReadIntoDF_Dr(self):
        totalDF = pd.DataFrame([])
        for i in range(self.num):
            oneDF = pd.read_csv(self.foldpath+self.datalist[i],encoding='latin1',engine='python')
            totalDF = totalDF.append([oneDF])
        self.totalDF = totalDF
        return totalDF
    def show_class_percentage_Dr(self):
        sns.set(font_scale=5)
        f,ax = plt.subplots(figsize=(60,20))
        ax.plot()
        sns.countplot(self.totalDF[' Label'],ax=ax)
    def move_label_remove_zero_feature_Dr(self,data):
        
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
        #keylist = list(sort_DF)
        #for i in keylist:
        #    if list(sort_DF[i]).count(0)==len(sort_DF):
        #       sort_DF.drop([i,],axis=1,inplace=True)
        #print("刪除都為0的欄位，資料型態為:{}".format(sort_DF.shape))
        return sort_DF
    def get_label_feature_Dr(self,data,slice_number):
        
        def augFeatures(data,time):
            data[time] = pd.to_datetime(data[time])
            data["month"] = data[time].dt.month
            data["date"] = data[time].dt.day
            data["hour"] = data[time].dt.hour
            data["minutes"] = data[time].dt.minute
            return data
        
        binaryDict = {}
        data.reset_index(drop=True,inplace=True)
        if self.label==True:
            containDF = pd.DataFrame([])
            class_sortDF = data.groupby(" Label")
            print("資料類別及數目總覽：\n",class_sortDF.size()) 

            
            for key in list(class_sortDF.groups):
                if len(class_sortDF.get_group(key))<slice_number:
                    a = pd.DataFrame(class_sortDF.get_group(key))
                else :
                    a = pd.DataFrame(class_sortDF.get_group(key)[:slice_number])#超過5000的話取5000筆
        
                #轉換為二分類
                if self.binary == True:
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
            containDF[" Flow Packets/s"] = containDF[" Flow Packets/s"].astype(float)
            containDF.drop(columns=[" Label"," Timestamp"," Destination IP"," Destination Port"," Source IP",
                   " Source Port","Flow ID","Flow Bytes/s"," Protocol",' Fwd Header Length.1',"External IP",#此行以後為經過ＲＦ判定為不重要的特徵，Eternal IP 為部份資料才有的特徵,' Fwd Header Length.1'是有label資料才有的
                " CWE Flag Count"," Bwd URG Flags","date","Bwd Avg Bulk Rate"," Bwd PSH Flags"," Flow Packets/s"," Fwd Avg Bulk Rate"," Fwd URG Flags",
                " Fwd Avg Packets/Bulk"," Bwd Avg Bytes/Bulk","Fwd Avg Bytes/Bulk"," Bwd Avg Packets/Bulk"],inplace=True)#Destination Port,source port,protocol 要做onehot
        else:
            containDF = data.copy()
            containDF["Flow Pkts/s"] = containDF["Flow Pkts/s"].astype(float)
            containDF = augFeatures(containDF,"Timestamp")
            containDF.drop(columns=["Label","Timestamp","Dst IP","Dst Port","Src IP",
                   "Src Port","Flow ID","Flow Byts/s","Protocol",
                   "Bwd Byts/b Avg","Fwd URG Flags","Bwd Blk Rate Avg","Bwd Pkts/b Avg","Bwd URG Flags","Fwd Blk Rate Avg",
                   "Fwd Pkts/b Avg","Bwd PSH Flags","Flow Pkts/s","ECE Flag Cnt","CWE Flag Count","RST Flag Cnt","Fwd Byts/b Avg"],inplace=True)#Destination Port,source port,protocol 要做onehot
        
        with pd.option_context('mode.use_inf_as_null', True):
            containDF.fillna(value=0.0,inplace=True)
        containDF=containDF.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))#最大最小標準化
        #print(list(containDF))
        containDF[list(containDF)] = containDF[list(containDF)].astype("float32")
        if self.label==True:
            return l,containDF
        
        else :
            features=containDF.values
            print("特徵的形狀為：{}".format(features.shape))
            return features
