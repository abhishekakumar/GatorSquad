import sys, SimpleXMLRPCServer,getopt, pickle, time, threading, xmlrpclib, unittest
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM
from xmlrpclib import Binary

class DiskPredictionServer:
    def __init__(self):
       self.in_dim = 19
       out_dim = 1
       hidden_neurons = 300
       self.model = Sequential()
       self.model.add(LSTM(output_dim=hidden_neurons,input_dim = self.in_dim,return_sequences=False))
       self.model.add(Dropout(0.5))
       self.model.add(Dense(output_dim=out_dim,input_dim=300,activation='linear'))
       self.model.compile(loss="binary_crossentropy", optimizer='adam',class_mode="binary")
       
    def getPrediction(self,value):
        print "Inside the server"
        inputData=pickle.loads(value.data)
        modelName=inputData['model']
        yearPrediction=inputData['year']
        monthPrediction=inputData['month']
        dayPrediction=inputData['day']
        val,testVal=self.getInputVector(yearPrediction,monthPrediction,dayPrediction,modelName)
        print "INput Vector:",val
        self.model.load_weights("/home/vyassu/GatorSquad/Weights/"+str(yearPrediction)+"/"+
                    str(monthPrediction)+"/"+str(modelName)+"_my_model_weights.h5")
        predicted = self.model.predict_classes(val)
        #score, acc = self.model.evaluate(val, testVal,show_accuracy=True)
        print(" THE VALUE Predicted:",predicted[0][0])
        return Binary(pickle.dumps({'predicted':predicted[0][0]}))    #,'score':score,'accuracy':acc}))

    def getInputVector(self,year,month,day,model):
        inputFilePath = '/hadoop/elephas/Output/'+str(year)+"/"+str(month)+"/"+str(model)+".csv"
        featuresList,labelList=[],[]
        tempData = pd.read_csv(filepath_or_buffer=inputFilePath,usecols=       	['smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_9_raw',
'smart_10_raw','smart_12_raw','smart_184_raw','smart_187_raw','smart_188_raw','smart_189_raw',
'smart_190_raw','smart_197_raw','smart_198_raw','smart_199_raw','smart_240_raw','smart_241_raw','smart_242_raw'])
	tempLabel = pd.read_csv(filepath_or_buffer=inputFilePath,usecols=['failure'])
        featuresList.append(tempData.iloc[int(day):int(day)+30].as_matrix())
        labelList.append(tempLabel.iloc[int(day)].as_matrix())
        if labelList[0] == 1:
	   labelList[0] = 0
	else:
           labelList[0] = 1
        featuresNPArray = np.array(featuresList)
	labelNPArray = np.array(labelList)
	return featuresNPArray,labelNPArray

def main():
  diskserve(int(sys.argv[1]))

# Start the xmlrpc server
def diskserve(port):
  disk_server = SimpleXMLRPCServer.SimpleXMLRPCServer(('', port),allow_none=True)
  disk_server.register_introspection_functions()
  disk = DiskPredictionServer()
  disk_server.register_function(disk.getPrediction)
  disk_server.serve_forever()

if __name__ == '__main__':
   main()
