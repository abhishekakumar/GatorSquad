from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout,TimeDistributedDense
from keras.layers.recurrent import LSTM,JZS1
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from random import random
import numpy as np
import pandas as pd
import readfile as rd
import pickle
from time import time
import datetime
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc

class cloudLSTM:

    def __init__(self):
       self.in_dim = 19
       self.n_prev=25
       self.future=34
       out_dim = 1
       hidden_neurons = 300
       self.model = Sequential()
       
       #self.model.add(Embedding(256, 1))
       self.model.add(LSTM(output_dim=hidden_neurons,input_dim = self.in_dim,return_sequences=False))
       self.model.add(Dropout(0.5))
       #self.model.add(LSTM(215,215, return_sequences=True))
       #self.model.add(Dropout(0.2))
       #self.model.add(Dense(input_dim=300, output_dim=500))
       self.model.add(Dense(output_dim=out_dim,input_dim=300,activation='linear'))
       #self.model.compile(loss="mean_squared_error", optimizer="rmsprop")
       self.model.compile(loss="binary_crossentropy", optimizer='adam',class_mode="binary")
       #self.model.compile(optimizer="sgd")
    
    def get_LSTM_Model(self):
        return self.model
    
    def write_file(self,data, filename='/home/user/Desktop/LOg.txt'):
        #filename='/home/user/Desktop/LOg.txt'
	f = open(filename, "ab+")
	f.write(str(data) + "\n")  
	f.close()

    def _load_data(self, data, label, n_prev = 30):
        """
        data should be pd.DataFrame()
        """
        self.n_prev=n_prev
        
        
        docX, docY = [],[]
        for i in range(len(data)-n_prev):
        	docX.append(data.iloc[i:i+n_prev].as_matrix())
        	docY.append(label.iloc[i+n_prev].as_matrix())
    	alsX = np.array(docX)
    	alsY = np.array(docY)
    	return alsX, alsY
   
    def load_weights_lstm(self, filepath):
        self.model.load_weights(filepath)
     
    def train_test_split(self,df, label, test_size=0.2,n_prev=30):
        ntrn = int(round(len(df) * (1 - test_size)))
        print('Number of training sample', ntrn)
        print('Number of testing sample', len(df) - ntrn)
	
        noOfOnes=0
	failure_cases=[]
        x_train, y_train = self._load_data(df.iloc[0:ntrn], label.iloc[0:ntrn])
        for i in range(0,len(y_train),1):
            if int(y_train[i][0])==1:
               noOfOnes+=1
               failure_cases.append(i)
               y_train[i][0]=0
            else:
               y_train[i][0]=1
               
        print("No of Failures:",noOfOnes)
        print y_train
        x_train_temp=[]
        y_train_temp=[]
        for val in failure_cases:
            print(val)
            if val > self.future:
               for j in range((val-self.future),(val+1),1):
                   x_train_temp.append(x_train[j])
                   y_train_temp.append(y_train[j])
        
        x_train=np.array(x_train_temp)
        y_train=np.array(y_train_temp)
        x_test, y_test = self._load_data(df.iloc[ntrn:], label.iloc[ntrn:])  
        for i in range(0,len(y_test),1):
            if int(y_test[i][0])==1:
               y_test[i][0]=0
            else:
               y_test[i][0]=1

        return (x_train, y_train), (x_test, y_test)
       
    def _train_test_split(self,df, label, test_size=0.2):
        """
        This just splits data to training and testing parts
        """
        ntrn = int(round(len(df) * (1 - test_size)))
        print('Number of training sample', ntrn)
        print('Number of testing sample', len(df) - ntrn)

        x_train, y_train = self._load_data(df.iloc[0:ntrn], label.iloc[0:ntrn])
        # converting the 2D input data to 3D for LSTM Input
        #x_train = x_train[:, np.newaxis, :]
        #y_train = self._load_data(label.iloc[0:ntrn])
        noOfOnes=0
        temp_sweights=[]
        failure_cases=[]
     
        for i in range(0,len(y_train),1):
            if int(y_train[i][0])==1:
               temp_sweights.append([0.5])
               noOfOnes+=1
               failure_cases.append(x_train[i])
            #self.write_file(x_train[i])
            else:
	       temp_sweights.append([0.5])
        k=0
        print("Current Length",str(len(x_train)))
        print("no of failure",noOfOnes)
     
        req_len=len(x_train)-(2*noOfOnes)
        print("No of New elements added",req_len)
        for j in range(0,req_len,1):
         if k < len(failure_cases):
              #x_train[len(x_train)+j]=failure_cases[k]
              x_train=np.append(x_train, np.resize(failure_cases[k],(1,self.n_prev,self.in_dim)),axis=0)
              self.write_file(x_train[i+1],filename='/home/user/Desktop/LOg_x_train.txt')
              y_train=np.append(y_train,[[1]],axis=0)
              self.write_file(y_train,filename='/home/user/Desktop/LOg_y_train.txt')
              k+=1
         else:
              k=0
              x_train=np.append(x_train, np.resize(failure_cases[k],(1,self.n_prev,self.in_dim)),axis=0)
              self.write_file("Data reset",filename='/home/user/Desktop/LOg_x_train.txt')
              y_train=np.append(y_train,[[1]],axis=0)
              self.write_file(y_train,filename='/home/user/Desktop/LOg_y_train.txt')
              #self.write_file("Data reset",filename='/home/user/Desktop/LOg_y_train.txt')
        v=0
        a=0
        for k in range(0,len(y_train),1):
         if int(y_train[k][0])==1:
            v+=1
         else:
            a+=1
        print("No of new ones:",v)
        print("No of new zeros:",a)

        x_test, y_test = self._load_data(df.iloc[ntrn:], label.iloc[ntrn:])
        #x_test = x_test[:, np.newaxis, :]
        #y_test = self._load_data(label.iloc[ntrn:])
        return (x_train, y_train), (x_test, y_test)
    
    
    def _prediction(self,data,label):
       (x_train, y_train), (x_test, y_test) = self.train_test_split(data,label)  # retrieve data
       print "reached out"
    def prediction(self,data,label):
     batchSize = 50
     (x_train, y_train), (x_test, y_test) = self.train_test_split(data.fillna(0),label.fillna(0))  # retrieve data
     #print x_train.shape
     #print y_train.shape
     
     #print x_train[76]
     #print x_train[77]
     print("new length of x_train;",str(len(x_train)))
     print("new length of y_train;",str(len(y_train)))
     #print noOfOnes
     #sam_weights= np.array(temp_sweights)
     #print sam_weights
     #testVal = pd.DataFrame(data=temp_sweights)
     #print testVal[0][1]

     #arr = []
     #for 
     #print(x_train)
     #print(y_train)
     #self.model.fit(x_train, y_train, batch_size=batchSize, nb_epoch=10,    	show_accuracy=True,shuffle=True)#,sample_weight=sam_weights)
     #self.model.save_weights("my_model_weights.h5",overwrite=True)
     self.model.load_weights("/home/user/Desktop/Cloud/Weights/my_model_weights.h5")
     #print("INPUT data")
     #print(x_test)
     count,count1=0,0
     for i in range(0,len(y_test)):
         if int(y_test[i][0])==1:
            count1=i
            y_test[i][0]=0
         else:
            y_test[i][0]=1
		
     for i in range(0,len(y_train)):
         if int(y_train[i][0])==0:
            count=i
     #print count
     #print count1
     #pd.DataFrame(y_test).plot()
     #val = np.resize(x_train[count],(1,self.n_prev,self.in_dim))
     #predicted = self.model.predict(val)
     #predicted1 = self.model.predict_classes(val)
     #score, acc = self.model.evaluate(x_test, y_test,show_accuracy=True)
     #print score
     #print acc
     #test_preds = self.model.predict_proba(x_test, verbose=0)
     #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, test_preds)
     #roc_auc = auc(false_positive_rate, true_positive_rate)
    
     #plt.title('Receiver Operating Characteristic')
     #plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
     #plt.legend(loc='lower right')
     #plt.plot([0,1],[0,1],'r--')
     #plt.xlim([-0.1,1.2])
     #plt.ylim([-0.1,1.2])
     #plt.ylabel('True Positive Rate')
     #plt.xlabel('False Positive Rate')
     #plt.show()
     #print(roc_auc)
     #print()
     print(str(y_train[count])+":::"+str(predicted1))
     self.write_file(predicted1,filename='/home/user/Desktop/Output.txt')
     success=0
     test_falsenegative=0
     test_falsepositive=0
     for a in range(0,100):
         val1= np.resize(x_test[a],(1,self.n_prev,self.in_dim))
         #print(y_test[a])
         #print(self.model.predict_classes(val1))
         pre=self.model.predict_classes(val1)
         if str(y_test[a][0])==str(pre[0][0]):
            success+=1
         elif(str(y_test[a][0])==1 and str(pre[0][0])==0):
            test_falsenegative+=1
         elif(str(y_test[a][0])==0 and str(pre[0][0])==1):
            test_falsepositive+=1
         value1="Actual Value:"+str(y_test[a])+" Predicted Value:"+str(pre)
         self.write_file(value1,filename='/home/user/Desktop/Output.txt')
     
     self.write_file("Training Data",filename='/home/user/Desktop/Output.txt')
     falsenegative=0
     falsepositive=0
     for a in range(0,100):
         val1= np.resize(x_train[a],(1,self.n_prev,self.in_dim))
         #print(y_test[a])
         #print(self.model.predict_classes(val1))
         pre=self.model.predict_classes(val1)
         value1="Actual Value:"+str(y_train[a])+" Predicted Value:"+str(pre)
         if str(y_train[a])==str(pre):
            success+=1
         elif(str(y_train[a][0])==1 and str(pre[0][0])==0):
            falsenegative+=1
         elif(str(y_train[a][0])==0 and str(pre[0][0])==1):
            falsepositive+=1
         self.write_file(value1,filename='/home/user/Desktop/Output.txt')
     #predicted2 = self.model.predict_classes(val1)
     print("output data")

     print("Test False negative:",test_falsenegative)
     print("Test False positive:",test_falsepositive)
     print("Training False negative:",falsenegative)
     print("Training False negative:",falsepositive)
     print("Success:",success)
     #predict=self.model.evaluate(x_test, show_accuracy=True)
     predicted1=self.model.predict_classes(x_test)
     df_confusion = pd.crosstab(y_test.flatten(), predicted1.flatten(), rownames=['Actual'], colnames=['Predicted'], margins=True)
     print df_confusion
     #pd.DataFrame(y_test).to_csv("CloudYTest_data.csv")
     #pd.DataFrame(predicted1).to_csv("CloudpredictedClass.csv")
     
     return


def main():
    testLSTM = cloudLSTM()
    #flowV = rd.generate_DataFrame("/Users/Abhishek/Courses/Cloud/Project/Data/2014/2014-01-*.csv")
    #flowV= rd.generate_DataFrame("/Users/Abhishek/Courses/Cloud/Project/Data/2014/2014-01-01.csv")
    flowV= rd.generate_DataFrame("/home/user/Desktop/Cloud/Test/2014/2014-05*.csv")
    testFeature = pd.read_csv(filepath_or_buffer="/home/user/Desktop/Cloud/Output/ST4000DM000.csv",usecols=['smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_9_raw','smart_10_raw','smart_12_raw',
 'smart_184_raw','smart_187_raw','smart_188_raw','smart_189_raw','smart_190_raw','smart_197_raw','smart_198_raw',
 'smart_199_raw','smart_240_raw','smart_241_raw','smart_242_raw'])
#['smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_240_raw'])
    testLabel = pd.read_csv(filepath_or_buffer="/home/user/Desktop/Cloud/Output/ST4000DM000.csv",usecols=['failure'])
    #print testFeature
    #print testLabel
    #test = pd.DataFrame(data=flowV['ST31500341AS'],columns=['serial_number','smart_1_raw','smart_5_raw','smart_9_raw',
							   #'smart_194_raw','smart_197_raw','failure'])
    
    #test_data = test.groupby('serial_number')
    #test_arr,test_arr1 = [],[]
    #pdata1 = pd.DataFrame()
    #pdata2 = pd.DataFrame()
    #for name,val in test_data:
        #test_data = pd.DataFrame(data=val,columns=['smart_1_raw','smart_5_raw','smart_9_raw',
                                                          #'smart_194_raw','smart_197_raw'])
        #test_label = pd.DataFrame(data=val,columns=['failure'])
        #test_arr.append(test_data.as_matrix())
        #test_arr1.append(test_label.as_matrix())
        #pdata1 = pdata1.append(test_data,ignore_index=True,verify_integrity=False)
        #pdata2 = pdata2.append(test_label,ignore_index=True,verify_integrity=False)

    #XData = np.array(test_arr)
    #print XData.shape
    #print pdata1
    #print XData
    #print pdata2
    #pdata2 = pd.DataFrame({"b":flowb,"c":flowb,"r":flowb,"q":flowb,"t":flowb})
    #pdata2 = pd.DataFrame({"b":flowb})
    #pdata1 = pd.DataFrame({"a":flowa,"b":flowc,"x":flowd,"y":flowe,"z":flowf})
    testLSTM.prediction(testFeature,testLabel)


if __name__ == '__main__':
    print ("Start time:",datetime.datetime.now().time())
    main()
    print ("End time:",datetime.datetime.now().time())
