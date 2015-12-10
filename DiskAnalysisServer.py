import sys, SimpleXMLRPCServer,getopt, pickle, time, threading, xmlrpclib, unittest
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM
from xmlrpclib import Binary

def getSMARTParameters():
    filePointer = open(os.environ.get('GATOR_SQUAD_HOME')+"Code/configFile.txt", "r+")
    SMARTParameterList=[]
    for line in filePointer:
        line = line.replace("\n","")
        SMARTParameterList= line.split(",")
    return SMARTParameterList


class DiskPredictionServer:
    # Initialzing the Class with default arguments for the LSTM Model
    def __init__(self):
       self.in_dim = len(getSMARTParameters())
       out_dim = 1
       neurons = 300
       self.model = Sequential()
       self.model.add(LSTM(output_dim=neurons,input_dim = self.in_dim,return_sequences=False))
       self.model.add(Dropout(0.5))
       self.model.add(Dense(output_dim=out_dim,input_dim=neurons,activation='linear'))
       self.model.compile(loss="binary_crossentropy", optimizer='adam',class_mode="binary")
       
    def getPrediction(self,value):
        # Receive the input data and convert from Binary to Dict type
        inputData=pickle.loads(value.data)

        #Retrieving the various paramaters for Loading the Model Weights
        modelName=inputData['model']
        yearPrediction=inputData['year']
        monthPrediction=inputData['month']
        dayPrediction=inputData['day']

        # Retrieving the 3D input vector from the Model data
        val,testVal=self.getInputVector(yearPrediction,monthPrediction,dayPrediction,modelName)

        # Loading the model weights stored after the training has been done
        self.model.load_weights("/home/vyassu/GatorSquad/Weights/"+str(yearPrediction)+"/"+
                    str(monthPrediction)+"/"+str(modelName)+"_my_model_weights.h5")

        # Predicting the status of the disk data for the given day
        predicted = self.model.predict_classes(val)
        #score, acc = self.model.evaluate(val, testVal,show_accuracy=True)
        return Binary(pickle.dumps({'predicted':predicted[0][0]}))

    def getInputVector(self,year,month,day,model):
        # Retrieving the SMART Parameters from the configFile
        SMARTParms= getSMARTParameters()

        # Retrieving the Input data file for the inputed model for the request date
        inputFilePath = '/hadoop/elephas/Output/'+str(year)+"/"+str(month)+"/"+str(model)+".csv"

        # Variables to store the Features and Label
        featuresList,labelList=[],[]

        # Reading the CSV file and converting the data into dataframes
        tempData = pd.read_csv(filepath_or_buffer=inputFilePath,usecols=SMARTParms)
        tempLabel = pd.read_csv(filepath_or_buffer=inputFilePath,usecols=['failure'])

        # Converting Dataframes into Lists
        featuresList.append(tempData.iloc[int(day):int(day)+30].as_matrix())
        labelList.append(tempLabel.iloc[int(day)].as_matrix())

        # Condition to check if the given data was failure or success
        if labelList[0] == 1:
            labelList[0] = 0
        else:
           labelList[0] = 1

        # Converting List into NPArray for data features and labels
        featuresNPArray = np.array(featuresList)
        labelNPArray = np.array(labelList)
        return featuresNPArray,labelNPArray

def main():
  diskserve(int(sys.argv[1]))

# Start the xmlrpc server
def diskserve(port):
  disk_server = SimpleXMLRPCServer.SimpleXMLRPCServer(('', port),allow_none=True)
  disk_server.register_introspection_functions()
  # Initializing the Server
  disk = DiskPredictionServer()
  disk_server.register_function(disk.getPrediction)
  disk_server.serve_forever()

if __name__ == '__main__':
   main()
