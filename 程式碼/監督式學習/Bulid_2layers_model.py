from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed,BatchNormalization,Bidirectional
import numpy as np
from keras.models import Model
from keras.layers import Input
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from ProcessBar import ShowProcess
class seq2seq:
    def __init__(self,x_train,y_train,x_test,y_test,n_units,binary=False):
        self.n_features = x_train.shape[2]
        self.n_classes  = y_test.shape[2]
        self.n_units = n_units
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.binary = binary
    def model_build(self):
        encoder_input = Input(shape=(None,self.n_features))
        encoder_lstm =Bidirectional(LSTM(128,return_sequences = True))
        lstm1 = encoder_lstm(encoder_input)
        encoder_lstm2 =LSTM(64,return_state = True)
        lstm2,state_h,state_c = encoder_lstm2(lstm1)
        state_h = BatchNormalization()(state_h)
        state_c = BatchNormalization()(state_c)
        initial = [state_h,state_c]
        decoder_input = Input(shape=(None,self.n_features))
        decoder_lstm_1= LSTM(64,return_sequences=True)
        decoder_lstm_1_out = decoder_lstm_1(decoder_input,initial_state=initial)
        decoder_lstm_2 = LSTM(128,return_sequences = True,return_state=True)
        decoder_lstm_2_out,_,_ = decoder_lstm_2(decoder_lstm_1_out)
        if self.binary ==False:
            decoder_dense = Dense(units = self.n_classes,activation="softmax")
        else:
            decoder_dense = Dense(2,activation = "softmax")
        decoder_output = decoder_dense(decoder_lstm_2_out)
        total_model = Model([encoder_input,decoder_input],decoder_output)
        self.total_model = total_model
        
        #define predict_encoder
        encoder_model = Model(encoder_input,initial)
        self.encoder_model = encoder_model
        #define predict_encoder
        decoder_initial_state_h = Input(shape=(64,))
        decoder_initial_state_c = Input(shape=(64,))
        decoder_state_inputs = [decoder_initial_state_h,decoder_initial_state_c]
        #decoder_lstm_2= LSTM(n_units,return_sequences = True,return_state = True)
        decoder_out1 = decoder_lstm_1(decoder_input,initial_state = decoder_state_inputs)
        decoder_out2,decoder_state_h,decoder_state_c = decoder_lstm_2(decoder_out1)
        new_state = [decoder_state_h,decoder_state_c]
        decoder_out = decoder_dense(decoder_out2)
        decoder_model = Model([decoder_input]+decoder_state_inputs,[decoder_out]+new_state)
        self.decoder_model = decoder_model
        print(total_model.summary())
        print(encoder_model.summary())
        print(decoder_model.summary())
        return total_model,encoder_model,decoder_model

    def train(self,delaydata,epoch = 10):
        self.total_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
        history = self.total_model.fit([self.x_train,delaydata],self.y_train,epochs=epoch)
        return history
    def predict_sequence(self,mapdict,delay_x_test):
        target = delay_x_test[0].reshape(1,1,self.n_features)
        print("\ntarget's shape = ",target.shape)
        c = 0
        # target = np.zeros(self.n_features).reshape(1,1,self.n_features)
        output = list()
        prediction = []
        labels = []
        score_list = []
        raw_p = []
        raw_l = []
        
        process_bar = ShowProcess(len(self.x_test),infoDone="預測完成！\n")
        
        for i in self.x_test:
            process_bar.show_process()
            i = i.reshape(1,1,-1)
            state = self.encoder_model.predict(i)
            y_hat,state_h,state_c = self.decoder_model.predict([target]+state)
            output.append(y_hat)
            #target = i
            if c <=delay_x_test.shape[0]-2:
                c+=1
                target = delay_x_test[c].reshape(1,1,self.n_features)
            else:
                pass
        pre = np.array(output)
        total = len(pre)
        correct = 0
        process_bar2 = ShowProcess(len(pre),infoDone="準確率計算完成！\n")
        print("計算準確率 ....")
        for i in range(len(pre)):
            process_bar2.show_process()
            p = np.argmax(pre[i][0])
            r = np.argmax(self.y_test[i])#被改過
            #找閥值
            if p ==0:
                score = np.min(pre[i][0])
            else :
                score = np.max(pre[i][0])
            if p==r:
                correct+=1
            else:
                pass
            raw_p.append(p)
            raw_l.append(r)
            prediction.append(mapdict[p])
            labels.append(mapdict[r])
            score_list.append(score)
        print("Accuracy is :%.5f"%(correct/total))
        acc = correct/total
        self.prediction,self.labels = np.array(prediction),np.array(labels)
        return np.array(prediction),np.array(labels),np.array(score_list),raw_p,raw_l,acc

    def AUC_plot(self,score,datasetname):
        LaDF = pd.DataFrame(self.labels)
        LaDF.replace({"Malware":1,"BENIGN":0},inplace=True)
        #preDF = pd.DataFrame(self.prediction).replace({"Malware":1,"BENIGN":0})
        fpr, tpr, threshold = metrics.roc_curve(LaDF.values,score)
        auc = metrics.roc_auc_score(LaDF.values,score)
        return fpr,tpr,auc
        # plt.plot(fpr,tpr,label="dataset: "+datasetname+", auc="+str(auc))
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.ylim(0,1.1)
        # plt.legend(loc = "upper right")
        # plt.savefig("/home/jim/Desktop/實驗紀錄")


# class MLP(seq2seq):
#     def __init__(self,x_train,y_train,x_test,y_test,n_units):
#         super().__init__(self,x_train,y_train,x_test,y_test,n_units)
#     def model_build()
        