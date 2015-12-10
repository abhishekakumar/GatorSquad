from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM
from random import random
import numpy as np
import pandas as pd
import ReadInputCSVFile as rd
from time import time
import datetime
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc

def getSMARTParameters():
    filePointer = open(os.environ.get('GATOR_SQUAD_HOME')+"Code/configFile.txt", "r+")
    SMARTParameterList=[]
    for line in filePointer:
        line = line.replace("\n","")
        SMARTParameterList= line.split(",")
    return SMARTParameterList

class cloudLSTM:
    # Initializing the LSTM Model
    def __init__(self,parms,timeSteps):
       self.in_dim = len(parms)                                                #Setting the Input Dimension according to the number of SMART parameters
       self.n_prev=0                                                           # Initialing the Number of prev to 0
       self.future=timeSteps                                                   # No of Timesteps
       out_dim = 1                                                             # Variable to store the Output dimension
       neurons = 300                                                           # No of Neurons in the hidden layer
       self.model = Sequential()                                               # Initializing the sequential LSTM model
       self.model.add(LSTM(output_dim=neurons,input_dim = self.in_dim,return_sequences=False))
       self.model.add(Dropout(0.5))
       self.model.add(Dense(output_dim=out_dim,input_dim=neurons,activation='linear'))
       self.model.compile(loss="binary_crossentropy", optimizer='adam',class_mode="binary")

    # Method to return the LSTM Model
    def get_LSTM_Model(self):
        return self.model

    # Method to convert Dataframes into NPArray for Keras LSTM Model
    def _load_data(self, data, label, n_prev = 30):
        self.n_prev=n_prev
        featureList, labelList = [],[]

        # Loop to traverse the Dataframe to convert to List of Matrices
        for i in range(len(data)-n_prev):
        	featureList.append(data.iloc[i:i+n_prev].as_matrix())
        	labelList.append(label.iloc[i+n_prev].as_matrix())

        #Coverting List of Matrices into 3D NPArray
    	featureNPArray = np.array(featureList)
    	labelNPArray = np.array(labelList)

    	return featureNPArray, labelNPArray
     
    def train_test_split(self,data, label, test_size=0.2,n_prev=30):
        ntrn = int(round(len(data) * (1 - test_size)))
        print('Number of training sample', ntrn)
        print('Number of testing sample', len(data) - ntrn)
	
        noOfOnes=0                                                 # Initializing the counter to calculate the number of Failures
        failure_cases=[]                                           # List to store the Failure cases

        # Function invoked to convert Dataframe to NPArray
        feature_train, label_train = self._load_data(data.iloc[0:ntrn], label.iloc[0:ntrn])

        # Loop to traverse the NPArray to find the failure cases
        for i in range(len(feature_train)):
            #Condition to check if the given disk data failed
            if int(label_train[i][0])==1:
               noOfOnes+=1
               failure_cases.append(i)
               label_train[i][0]=0                  # Changing all Failure cases status from 1 to 0 for better model weight convergence
            else:
               label_train[i][0]=1                  # Changing all Success cases status from 0 to 1 for better model weight convergence
               
        print("No of Failures:",noOfOnes)

        # Begining Undersampling of Training data
        # Initializing Temporary List varaiables to store the Undersampled data
        feature_train_temp=[]
        label_train_temp=[]

        # Loop to traverse the Failure list
        for val in failure_cases:
            # Condition to pick success data from the input Data list
            if val > self.future:
               # Loop to traverse the entire INput Data frame for Success data
               for j in range((val-self.future),(val+1),1):
                   feature_train_temp.append(feature_train[j])
                   label_train_temp.append(label_train[j])

        # Copying the Undersampled NPArray back to the original variable
        feature_train=np.array(feature_train_temp)
        label_train=np.array(label_train_temp)

        # Function invoked to convert Dataframe to NPArray
        feature_test, label_test = self._load_data(data.iloc[ntrn:], label.iloc[ntrn:])

        # Loop to traverse the Test dataframes
        for i in range(0,len(label_test),1):
            #Condition to check if the given disk data failed
            if int(label_test[i][0])==1:
               label_test[i][0]=0                               # Changing all Failure cases status from 1 to 0 for better model weight convergence
            else:
               label_test[i][0]=1                               # Changing all Success cases status from 0 to 1 for better model weight convergence

        return (feature_train, label_train), (feature_test, label_test)

    # Function to fit the training model and generate the model weights
    def prediction(self,data,label):
        batchSize = 50

        # Method to convert Dataframes to NPArray
        (feature_train, label_train), (feature_test, label_test) = self.train_test_split(data.fillna(0),label.fillna(0))  # retrieve data

        print("new length of undersample;",str(len(feature_train)))

        if len(feature_train) == 0:
            print("Training cannot proceed as there are no failure cases in the Test data")
            return

        # Training the LSTM Model
        self.model.fit(feature_train, label_train, batch_size=batchSize, nb_epoch=10,show_accuracy=True,shuffle=True)

        # Saving the Model weights in the current working directory
        self.model.save_weights("my_model_weights.h5",overwrite=True)
        #self.model.load_weights("/home/user/Desktop/Cloud/Weights/my_model_weights.h5")

        # Testing the model with the test data
        predicted = self.model.predict(feature_test)
        actualpredictVal = self.model.predict_classes(feature_test)

        # Retrieving the Score and Accuracy of the Model trained
        score, acc = self.model.evaluate(feature_test, label_test,show_accuracy=True)
        print("The model score:",score)
        print("The model accuracy:",accuracy)

        # Creating the ROC Table
        df_confusion = pd.crosstab(label_test.flatten(), actualpredictVal.flatten(), rownames=['Actual'], colnames=['Predicted'], margins=True)
        print("The ROC Table for the LSTM Model:")
        print df_confusion



def main():
    timeSteps=sys.argv[1]
    modelName=sys.argv[2]
    SMARTparms= getSMARTParameters()
    testLSTM = cloudLSTM(SMARTparms,30)
    rd.generate_DataFrame("/home/user/Desktop/Cloud/Test/2014/2014-05*.csv")
    diskModelFilePath= "/home/user/Desktop/Cloud/Output/"+modelName+".csv"
    testFeature = pd.read_csv(filepath_or_buffer=diskModelFilePath,usecols=SMARTparms)
    testLabel = pd.read_csv(filepath_or_buffer=diskModelFilePath,usecols=['failure'])
    testLSTM.prediction(testFeature,testLabel)


if __name__ == '__main__':
    print ("Start time:",datetime.datetime.now().time())
    main()
    print ("End time:",datetime.datetime.now().time())
