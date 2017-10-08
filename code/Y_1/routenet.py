import numpy as np
np.random.seed(2016)
import os,sys,gc
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import math
import random
random.seed(1)
import pandas
from collections import OrderedDict
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge
from keras.models import Model,model_from_json
from keras.optimizers import Adadelta,Adagrad
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

hist = []

def gendata():
    train_x = []
    train_y = []
    for i in range(200):
        x1=int(random.random() * 100)
        x2=int(random.random() * 100)
        y = x1 + x2
        train_x.append([x1,x2])
        train_y.append(y)
        x1=2+round(random.random()*3,3)
        x2=2+round(random.random()*3,3)
        y = x1 * x2  
        train_x.append([x1,x2])
        train_y.append(y)      
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print train_x[:10,:]
    print train_y[0:10]
    return train_x,train_y

def fit(modellist,modelmeta,X,Y,X_test = None,Y_test = None,ep = 1, mutant = False,batch_size=1,log = False):
    # modelmeta.trainable = False
    for e in range(ep):
        routetrain = modelmeta.predict(X)
        meta_label = routetrain.copy().reshape((Y.shape[0],len(modellist)))
        if mutant:
            routetrain += (np.random.random(routetrain.shape) - 0.5)/5
        #print routetrain[0:10]
        routetrain = np.array(map(lambda x: x.argmax(),routetrain)).reshape(-1,1)
        # train&update
        for modelseq in range(len(modellist)):
            index = np.where(routetrain == modelseq)[0]
            if index.shape[0] == 0:
                continue
            feat_np = []
            for x in X:
                feat_np.append(x[index])
            label_np = Y[index]
            #print meta_label[index, modelseq][0:10],meta_label[index, modelseq].shape
            if log:
                print 'modelnum: '+str(modelseq),label_np.shape  
            his = modellist[modelseq].fit(feat_np,label_np,nb_epoch=1, batch_size=batch_size)
            #his = modellist[modelseq].fit(feat_np,label_np,nb_epoch=1, batch_size=batch_size,sample_weight = meta_label[index, modelseq])
            pred = modellist[modelseq].predict(feat_np, batch_size=batch_size)
            loss = abs(label_np - pred.reshape(label_np.shape))/(label_np+0.00001)
            #print 'loss:',loss[:10]
            for j in range(len(loss)):
                meta_label[index[j], modelseq] = 1.0 - loss[j]
        meta_label = meta_label.clip(0,1)
        #print meta_label[0:10]

        # update meta 
        if log:
            print 'modelmeta:'
        modelmeta.fit(X,meta_label,nb_epoch=1, batch_size=batch_size)

        # valid
        '''
        if X_test_feat is not None:
            routetest = modelmeta.predict(X_test)
            routetest = np.array(map(lambda x: x.argmax(),routetest)).reshape(-1,1)
            for modelseq in range(len(modellist)):
                index = np.where(routetest == modelseq)[0]
                feat_np = []
                for x in X_test:
                    feat_np.append(x[index])
                label_np = Y[index].reshape(-1,1)
                res = modellist[modelseq].predict(feat_np,batch_size=500)
        '''
        Y_test_pred = np.zeros((Y_test.shape[0]))
        if X_test_feat is not None:
            routetest = modelmeta.predict(X_test)
            if log:
                print routetest[0:10]
            routetest = np.array(map(lambda x: x.argmax(),routetest)).reshape(-1,1)
            for modelseq in range(len(modellist)):
                index = np.where(routetest == modelseq)[0]
                feat_np = []
                if index.shape[0] == 0:
                    continue
                for x in X_test:
                    feat_np.append(x[index])
                res = modellist[modelseq].predict(feat_np,batch_size=500)
                for j in range(len(res)):
                    Y_test_pred[index[j]] = res[j]
            if log:
                print Y_test[:10]
                print Y_test_pred[:10]
            print 'loss',np.average(np.abs(Y_test-Y_test_pred)/Y_test)
            hist.append(np.average(np.abs(Y_test-Y_test_pred)/Y_test))
 
train_x,train_y = gendata()

X_train_feat = train_x[:-50,:]
Y_train_score = train_y[:-50]
X_test_feat = train_x[-50:,:]
Y_test_score = train_y[-50:]




# 1
input_1=Input(shape=(2,), dtype='float32')
x = BatchNormalization()(input_1)
x1 = Dense(100, activation='sigmoid')(x)
x2 = Dense(100)(x)
x2 = PReLU()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)

x1 = Dense(50, activation='sigmoid')(x)
x2 = Dense(50)(x)
x2 = PReLU()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)


out = Dense(1, activation='linear')(x)
model1 = Model(input=[input_1], output=out)
model1.compile(optimizer='adam',
              loss='mse',
              metrics=[])
              

# 2
input_1=Input(shape=(2,), dtype='float32')
x = BatchNormalization()(input_1)
x1 = Dense(100, activation='sigmoid')(x)
x2 = Dense(100)(x)
x2 = PReLU()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)

x1 = Dense(50, activation='sigmoid')(x)
x2 = Dense(50)(x)
x2 = PReLU()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)


out = Dense(1, activation='linear')(x)
model2 = Model(input=[input_1], output=out)
model2.compile(optimizer='adam',
              loss='mse',
              metrics=[])

#modellist = [model1]
modellist = [model1,model2]

# meta
input_1=Input(shape=(2,), dtype='float32')
x = BatchNormalization()(input_1)
x1 = Dense(40, activation='sigmoid')(x)
x2 = Dense(40)(x)
x2 = PReLU()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)

x1 = Dense(20, activation='sigmoid')(x)
x2 = Dense(20)(x)
x2 = PReLU()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)

out = Dense(len(modellist), activation='sigmoid')(x)
modelmeta = Model(input=[input_1], output=out)
modelmeta.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=[])
#modelmeta.summary()

 
#print X_train_feat.shape,Y_train_score.shape
fit(modellist,modelmeta,[X_train_feat],Y_train_score,X_test = [X_test_feat],Y_test = Y_test_score,ep = 80, mutant = True,batch_size=10)
print hist


    
