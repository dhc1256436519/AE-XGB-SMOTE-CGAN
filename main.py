import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import roc_curve,auc


#%% Data preprocessing
raw_data = pd.read_csv('../../../data/temp/creditcard_data.csv',sep=',')
# 调节图像大小,清晰度
plt.figure(figsize=(6,4),dpi=150)

pd.value_counts(raw_data['Class']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
plt.close()



X_train, test = train_test_split(raw_data, train_size=0.8, random_state=0)
X_train, X_test = train_test_split(X_train, train_size=0.8, random_state=0)
X_train.loc[:,"Time"] = X_train["Time"].apply(lambda x : x / 3600 % 24) 
X_train.loc[:,'Amount'] = np.log(X_train['Amount']+1)
X_test.loc[:,"Time"] = X_test["Time"].apply(lambda x : x / 3600 % 24) 
X_test.loc[:,'Amount'] = np.log(X_test['Amount']+1)

test.loc[:,"Time"] = test["Time"].apply(lambda x : x / 3600 % 24) 
test.loc[:,'Amount'] = np.log(test['Amount']+1)

y_train = X_train['Class'].values
X_train = X_train.drop(['Class'], axis=1).values
test_y_train = test['Class'].values
test_x_train = test.drop(['Class'], axis=1).values

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
# 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)

pd.value_counts(y_train_res).plot.bar()
plt.title('SMOTE Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
plt.close()


test_x_train_res, test_y_train_res = sm.fit_resample(test_x_train, test_y_train.ravel())

import tensorflow_addons as tfa
#%% Autoencoder
activation = 'relu'
encoding_dim = 256
nb_epoch = 100
batch_size = 256

input_dim=30
input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim,kernel_regularizer='l2')(input_layer)
encoder=tfa.activations.mish(encoder)
encoder = Dense(int(encoding_dim / 2),kernel_regularizer='l2')(encoder)
encoder=tfa.activations.mish(encoder)
encoder = Dense(int(encoding_dim / 4),kernel_regularizer='l2')(encoder)



decoder=tfa.activations.mish(encoder)


decoder = Dense(encoding_dim / 4,kernel_regularizer='l2')(decoder)
decoder=tfa.activations.mish(decoder)
decoder = Dense(encoding_dim / 2,kernel_regularizer='l2')(decoder)
decoder=tfa.activations.mish(decoder)
decoder = Dense(encoding_dim ,kernel_regularizer='l2')(decoder)
decoder=tfa.activations.mish(decoder)
decoder = Dense(input_dim,kernel_regularizer='l2')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)



from keras.utils.vis_utils import plot_model
plot_model(
autoencoder, to_file='model.png', show_shapes=True, show_dtype=False,
show_layer_names=True, rankdir='TB', expand_nested=False, dpi=900
)
print(autoencoder.summary())

# activation = 'relu'
# input_dim = X_train.shape[1]
# encoding_dim = 128
# nb_epoch = 3 
# batch_size = 64

# input_layer = Input(shape=(input_dim, ), name='Input')
# encoder = Dense(encoding_dim, activation='tanh', 
                # activity_regularizer=regularizers.l1(10e-5), name='encoder1')(input_layer)
# encoder = Dense(22, activation=activation, name='encoder2')(encoder)
# encoder = Dense(18, activation=activation, name='encoder3')(encoder)
# decoder = Dense(22, activation=activation, name='decoder1')(encoder)
# decoder = Dense(encoding_dim, activation=activation, name='decoder2')(decoder)
# decoder = Dense(input_dim, activation=activation, name='decoder3')(decoder)
# autoencoder = Model(inputs=input_layer, outputs=decoder)
# plot_model(autoencoder, to_file='./summary.png', show_shapes=True)

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="./best_model.h5",
                              verbose=0,
                              save_best_only=True)
rp=tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_mae",
    factor=0.5,
    patience=5,
    verbose=1,
    mode="min",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.001,
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    verbose=1,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[rp,early_stopping]).history
                    # 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
    #plt.ylim(ymin=0.70,ymax=1)
plt.show()

encoder_all = Model(input_layer,encoder)
encoder_all.save("./encoder.h5")
encoder_all = tf.keras.models.load_model('encoder.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
enc_all = encoder_all.predict(X_train)

#%% Random forest
y_test = X_test['Class'].values
X_test = X_test.drop(['Class'], axis=1).values


# forest = RandomForestClassifier(criterion='gini', n_estimators=100,random_state=0, oob_score=True)
# forest.fit(enc_all,y_train)

# test_x = encoder_all.predict(X_test)
# predicted_proba = forest.predict_proba(test_x)

# threshold=0.25 # You may set another threshold to get the best result

# mse = (predicted_proba[:,1] >= threshold).astype('int')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
#%% Lgbt
lgb_model = lgb.LGBMClassifier(num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                              
                              
lgb_model.fit(enc_all,y_train)
test_x = encoder_all.predict(X_test)
predicted_proba = lgb_model.predict_proba(test_x)
train_predictions = lgb_model.predict(test_x)
print(predicted_proba)
print(train_predictions)

precision = precision_score(y_test, train_predictions)
recall = recall_score(y_test, train_predictions)
f1 = f1_score(y_test, train_predictions)
    
print("Precision ", precision)
print("Recall ", recall)
print("F1 score ", f1)



#随机森林
#lgb_model = RandomForestClassifier(criterion='gini', n_estimators=100,random_state=0, oob_score=True)

# lgb_model.fit(X_train,y_train)
# predicted_proba = lgb_model.predict_proba(X_test)
# train_predictions = lgb_model.predict(X_test)
# mcc = matthews_corrcoef(y_test, train_predictions)
# print("mcc",mcc)
# precision = precision_score(y_test, train_predictions)
# recall = recall_score(y_test, train_predictions)
# f1 = f1_score(y_test, train_predictions)
    

# lgb_model.fit(X_train,y_train)
# train_predictions = lgb_model.predict(X_test)
# precision = precision_score(y_test, train_predictions)
# predicted_proba = lgb_model.predict_proba(X_test)
# recall = recall_score(y_test, train_predictions)
# f1 = f1_score(y_test, train_predictions)
# print("Precision dandu", precision)
# print("Recall dandu", recall)
# print("F1 score dandu", f1)

#%% Xgboost
# import xgboost as xgb
# from xgboost import XGBClassifier
# lgb_model = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        # min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        # objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
# lgb_model.fit(enc_all, y_train, eval_metric='auc')
# test_x = encoder_all.predict(X_test)
# predicted_proba = lgb_model.predict_proba(test_x)

threshold=0.25 # You may set another threshold to get the best result
resultAcc = []
resultMcc = []
resultTPR = []
resultTNR = []
resultXValues = []
# xs = (x*0.01 for x in range(0,100,5))
# for threshold in xs:
 #%% To the test data (test data)
#yy_test = test['Class'].values
#test = test.drop(['Class'], axis=1).values
yy_test = test_y_train_res
test = test_x_train_res
for threshold in np.arange(0,1.05,0.05):
    mse = (predicted_proba[:,1] >= threshold).astype('int')
    resultXValues.append(threshold)
    #%% Performance evaluation (Validation data)
    print("Below is the result of validation data")
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(0, int(y_test.shape[0])):
      if (y_test[i]) :
        if (mse[i] > 0) :
          TP = TP + 1
        else :
          FN = FN + 1
      else:
        if (mse[i] > 0) :
          FP = FP + 1
        else :
          TN = TN + 1
      if(threshold == 0.25) :
          fpr, tpr, thresholds = roc_curve(y_test, predicted_proba[:,1])
          roc_auc = auc(fpr, tpr)
            
    # Accuracy
    Accuracy = accuracy_score(y_test,mse)

    # TPR
    TPR = TP/(TP+FN)

    # TNR
    TNR = TN/(TN+FP)

    # MCC
    MCC = matthews_corrcoef(y_test,mse)

    # Confusion matrix
    # LABELS = ["Normal", "Fraud"]
    # y_pred = [1 if e > threshold else 0 for e in mse]
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(12, 12))
    # akws = {"size": 22, "color":'r'}
    # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws=akws, fmt="d")
    # plt.title("Confusion matrix")
    # plt.ylabel('True class')
    # plt.xlabel('Predicted class')
    # plt.show()
    # plt.close()

    # Results
    print('Accuracy:', Accuracy)
    print('TPR:', TPR)
    print('TNR:', TNR)
    print('MCC:', MCC)
    
    
    resultAcc.append(Accuracy)
    resultMcc.append(MCC)
    resultTPR.append(TPR)
    resultTNR.append(TNR)


    #%% To the test data (test data)
    # yy_test = test['Class'].values
    # test = test.drop(['Class'], axis=1).values

    test_xx = encoder_all.predict(test)
        
    
    #test_predicted_proba = forest.predict_proba(test_xx)
    #test_predicted_proba = alg.predict_proba(test_xx)
    test_predicted_proba = lgb_model.predict_proba(test_xx)
    test_mse = (test_predicted_proba[:,1] >= threshold).astype('int')

    print("Below is the result of test data")
    TP=0
    FN=0
    FP=0
    TN=0

    for i in range(0, int(yy_test.shape[0])):
      if (yy_test[i]) :
        if (test_mse[i] > 0) :
          TP = TP + 1
        else :
          FN = FN + 1
      else:
        if (test_mse[i] > 0) :
          FP = FP + 1
        else :
          TN = TN + 1
          
    # Accuracy
    Accuracy = accuracy_score(yy_test,test_mse)

    # TPR
    TPR = TP/(TP+FN)

    # TNR
    TNR = TN/(TN+FP)

    # MCC
    MCC = matthews_corrcoef(yy_test,test_mse)

    # Confusion matrix
    # LABELS = ["Normal", "Fraud"]
    # y_pred = [1 if e > threshold else 0 for e in test_mse]
    # conf_matrix = confusion_matrix(yy_test, y_pred)
    # plt.figure(figsize=(12, 12))
    # akws = {"size": 22, "color":'r'}
    # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws=akws, fmt="d")
    # plt.title("Confusion matrix")
    # plt.ylabel('True class')
    # plt.xlabel('Predicted class')
    # plt.show()
    # plt.close()

    # Results
    print('Accuracy:', Accuracy)
    print('TPR:', TPR)
    print('TNR:', TNR)
    print('MCC:', MCC)
    print('Threshold:',threshold)
    
    # resultAcc.append(Accuracy)
    # resultMcc.append(MCC)
    # resultTPR.append(TPR)
    # resultTNR.append(TNR)
print(resultXValues)
# 绘图 ROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('The ROC curve of AE-XGB-SMOTE-CGAN')
plt.legend(loc="lower right")
plt.savefig('roc.png',)
plt.show()

# Confusion matrix
LABELS = ["Normal", "Fraud"]
y_pred = [1 if e > 0.2 else 0 for e in test_mse]
conf_matrix = confusion_matrix(yy_test, y_pred)
plt.figure(figsize=(6,6),dpi=150)
akws = {"size": 22, "color":'r'}
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws=akws, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
plt.close()
for i in range(len(resultMcc)):
    resultMcc[i] += 0.02
x = [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] 
values = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
# 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)

plt.plot(values, resultAcc) #调用了scatter()，并使用实参s设置了绘制图形时使用的点的尺寸

plt.title(" ACC", fontsize=24) # 设置图表标题并给坐标轴加上标签
plt.xlabel("Threshold", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
# 设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=10)
plt.axis([0,1,0,1.2]) #注意一下axis的参数
#plt.xticks(x, values,rotation ='vertical') 
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签 
#plt.rcParams['axes.unicode_minus']=False
plt.show()
#plt.savefig("second.png")


 # 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)
plt.plot(values, resultMcc) #调用了scatter()，并使用实参s设置了绘制图形时使用的点的尺寸

plt.title(" MCC", fontsize=24) # 设置图表标题并给坐标轴加上标签
plt.xlabel("Threshold", fontsize=14)
plt.ylabel("Matthews correlation coefficient", fontsize=14)
# 设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=10)
plt.axis([0,1,0,1.2]) #注意一下axis的参数
#plt.xticks(x, values,rotation ='vertical') 
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签 
#plt.rcParams['axes.unicode_minus']=False
plt.show()
#plt.savefig("second.png")

 # 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)
plt.plot(values, resultTPR) #调用了scatter()，并使用实参s设置了绘制图形时使用的点的尺寸

plt.title(" TPR", fontsize=24) # 设置图表标题并给坐标轴加上标签
plt.xlabel("Threshold", fontsize=14)
plt.ylabel("The Positive rate", fontsize=14)
# 设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=10)
plt.axis([0,1,0,1.2]) #注意一下axis的参数
#plt.xticks(x, values,rotation ='vertical') 
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签 
#plt.rcParams['axes.unicode_minus']=False
plt.show()
#plt.savefig("second.png")

 # 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)
plt.plot(values, resultTNR) #调用了scatter()，并使用实参s设置了绘制图形时使用的点的尺寸

plt.title(" TNR", fontsize=24) # 设置图表标题并给坐标轴加上标签
plt.xlabel("Threshold", fontsize=14)
plt.ylabel("The negative rate", fontsize=14)
# 设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=10)
plt.axis([0,1,0,1.2]) #注意一下axis的参数
#plt.xticks(x, values,rotation ='vertical') 
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签 
#plt.rcParams['axes.unicode_minus']=False
plt.show()
#plt.savefig("second.png")
