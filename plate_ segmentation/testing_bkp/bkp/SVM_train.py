#decoding: utf-8
#the direction of this file is same wiht image folders

import os
import numpy as np
import cv2 as cv
import pandas as pd
import logging
import time 

def ini():  
    TM=time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())
    LOG_FORMAT = "%(levelname)s -[:%(lineno)d]- %(message)s"
    #open(log_nm,'w').close()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.info('**************************Mason(%s)*******************'%(TM,))


def load_data(path_X,path_Y):
    filelist=os.listdir(path_X)# all files: name.jpg
    
    df=pd.read_table(path_Y,sep=":",header=None)
    df.columns=['X_','Y_']
    
    X_=[]
    Y_=[]
    
    for files in filelist:
        path_=os.path.join(path_X,files)
        img=cv.imread(path_,cv.IMREAD_GRAYSCALE)
        N=(np.array(img).shape[1])*(np.array(img).shape[0])

        img=np.reshape(np.array(img),N)
        X_.append(img.tolist())# X_train is a list; can not use extend

        filename=os.path.splitext(files)[0];#文件名
        label=df.loc[df['X_']==filename,'Y_']
        label=float(label)
        Y_.append(label) # a list
    
    logging.info('the shape of %s is: %s;and %s is:%s '%(path_X,np.array(X_).shape,path_Y,np.array(Y_).shape))
    return X_, Y_
                  
def statistic(Y_,Y_predict):

    N=len(Y_train_predict)
    k=0
    for i in range (0,N):
        if Y_[i]==Y_predict[i]:#return  a  bool
            k=k+1
    correct=k
    logging.info('correct ratio is:%s'%(correct*100.0/N))

    
    
if __name__ == '__main__':
    ini()
    
    path_X_train= './Data/Train/X_train/'
    path_Y_train='./Data/Train/Y_train.txt'

    path_X_test= './Data/Test/X_test/'
    path_Y_test='./Data/Test/Y_test.txt'
    
    X_train,Y_train=load_data(path_X_train,path_Y_train)
    X_test,Y_test=load_data(path_X_test,path_Y_test) #label Y_ that load for txt is int here

    logging.info('create a svm...')
    svm = cv.ml.SVM_create()
    
    logging.info('svm_params...')

    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    #svm.setType(cv.ml.SVM_EPS_SVR)
    svm.setGamma(1.0)
    svm.setC(5.0)
    #svm.setP(0.2)
    
    logging.info('trainning...')
    svm.train(np.array(X_train,dtype=np.float32),cv.ml.ROW_SAMPLE,np.array(Y_train,dtype=np.int64))

    logging.info('save svm model...')
    svm.save('./model/mdoel.m')
    
    logging.info('predict X_train...')
    Y_train_predict=svm.predict(np.array(X_train,dtype=np.float32))# the output is a tuple value is float
    Y_train_predict=list(Y_train_predict)[1]
    #print  Y_train_predict[]
    logging.info('Y_train:%s \n'%np.array(Y_train,dtype=np.float32)) #Y_train is  list
    logging.info('Y_train_predict:%s\n'%Y_train_predict)# Y_train_predict is list
    
    logging.info('statistic...')
    statistic(Y_train,Y_train_predict)

                  
    

'''
import cv2


#input dataset
def DataSet():
    #input X_train dataset
    X_train=[]
    path_X_train = sorted(list(paths.list_images('./Data/Train/X_train')))
    k=0
    for i in path_X_train:
        X_train[k]=cv.imread(i)
        k=k+1
    #input Y_train
    Y_train=[]
    path_Y_train = sorted(list(paths.list_images('./Data/Train/Y_train')))
    k=0
    for i in path_Y_train:
        Y_train[k]=cv.imread(i)
        k=k+1
    #input X_vali
    X_vali=[]
    path_X_vali = sorted(list(paths.list_images('./Data/Train/X_vali')))
    k=0
    for i in path_X_vali:
        X_vali[k]=cv.imread(i)
        k=k+1
    #input Y_vali
    Y_vali=[]
    path_Y_vali = sorted(list(paths.list_images('./Data/Train/Y_vali')))
    k=0
    for i in path_Y_vali:
        Y_vali[k]=cv.imread(i)
        k=k+1
    #input X_test
    X_test=[]
    path_X_test= sorted(list(paths.list_images('./Data/Test/X_test')))
    k=0
    for i in path_X_test:
        X_test[k]=cv.imread(i)
        k=k+1
    #input Y_test
    Y_test=[]
    path_Y_test= sorted(list(paths.list_images('./Data/Test/Y_test')))
    k=0
    for i in path_Y_test:
        Y_test[k]=cv.imread(i)
        k=k+1
return X_train, X_vali, X_test, Y_train, Y_vali, Y_train

def Parameters():

def statistic(Y_test_predict,Y_test):

    mask = Y_test==Y_test_predict
    correct = np.count_nonzero(mask)
    print(correct*100.0/Y_test_predict.size)



if __name__ == '__main__':
    X_train, X_vali, X_test, Y_train, Y_vali, Y_train=DataSet()
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
    svm = cv2.ml.SVM_create()
    #trainning
    svm.tarin(X_train,Y_train,params=svm_params)
    svm.save('./model/mdoel.m')
    Y_vali_predict=svm.predict(X_vali)
    Y_test_predict=svm.predict(X_test)
    statistic(Y_test_predict,Y_test)
'''
    

