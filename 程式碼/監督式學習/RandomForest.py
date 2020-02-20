from processing_ColumnDrop import DataPreprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


ISCX = DataPreprocessing(foldpath="/home/jim/MalwareData/ISCX/",num = 8,label=True,binary=False)
rawDF = ISCX.ReadIntoDF()
sortDF = ISCX.move_label_remove_zero_feature(rawDF)
label,features = ISCX.get_label_feature(sortDF,1000000)#超過500就捨棄
print("\n有{}個Label".format(len(label)))
print("特徵數目為：{}\n".format(features.shape[1]))

n_features = features.shape[1]
x_train,y_train,x_test,y_test,L_onehot = train_test_split(label = label,features = features,percent = 0.9)
n_classes = y_test.shape[1]
n_units = 32
datalist = [x_train,y_train,x_test,y_test]
namelist = ["x_train","y_train","x_test","y_test"]
dataDict = {}
featuredict = {}
for f in range(len(features.keys())):
    featuredict[f] = features.keys()[f]
print("featuredict test : feature[0] = ",featuredict[0])
X, y = x_train,y_train
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f),this feature's name is : %s" % (f , indices[f], importances[indices[f]],featuredict[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(20,4))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()