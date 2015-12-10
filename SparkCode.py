import LSTM_Latest as ls
import readfile as rd
import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers
import elephas.spark_model as sm

config = SparkConf().setAppName('DiskDetection_App').setMaster('local[6]')  #[4] indicates the number of threads on 											 the master node
sc = SparkContext(conf=config)
print "Going to Initialize the LSTM model"
lstm = ls.cloudLSTM()
print "Initialized the Model"
lstmModel = lstm.get_LSTM_Model()

flowV= rd.generate_DataFrame("/home/vyassu/GatorSquad/Input/csv/2014/2014-11*.csv")
#test = pd.DataFrame(data=flowV['ST1500DL003'],columns=['serial_number','smart_1_raw','smart_5_raw','smart_9_raw',
							   #'smart_194_raw','smart_197_raw','failure'])
testFeature = pd.read_csv(filepath_or_buffer="/hadoop/elephas/Output/ST4000DM000.csv",usecols=['smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_9_raw','smart_10_raw','smart_12_raw',
 'smart_184_raw','smart_187_raw','smart_188_raw','smart_189_raw','smart_190_raw','smart_197_raw','smart_198_raw',
 'smart_199_raw','smart_240_raw','smart_241_raw','smart_242_raw'])
#['smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_240_raw'])
testLabel = pd.read_csv(filepath_or_buffer="/hadoop/elephas/Output/ST4000DM000.csv",usecols=['failure'])

#test_data = test.groupby('serial_number')
#pdata1 = pd.DataFrame()
#pdata2 = pd.DataFrame()
#for name,val in test_data:
    #test_data = pd.DataFrame(data=val,columns=['smart_1_raw','smart_5_raw','smart_9_raw',
                                                          #'smart_194_raw','smart_197_raw'])
    #test_label = pd.DataFrame(data=val,columns=['failure'])
    #pdata1 = pdata1.append(test_data,ignore_index=True,verify_integrity=False)
    #pdata2 = pdata2.append(test_label,ignore_index=True,verify_integrity=False)


(x_train, y_train), (x_test, y_test) = lstm.train_test_split(testFeature.fillna(0),testLabel.fillna(0),0.2,30)

print("Training Data")
#print(x_train)
#print(y_train)
#sweights=lstm.get_sample_weight(y_train)
count1=0
for i in range(0,len(y_test)):
     if int(y_test[i][0])==1:
        count1=i
count2=0
count=0	
for i in range(0,len(y_train)):
     if int(y_train[i][0])==1:
        count=i
        count2+=1
print count
print count1
print count2

#sigmoid = elephas_optimizers.SGD()
adam = elephas_optimizers.Adam()
print "Adam Optimizer initialized"
rddataset = to_simple_rdd(sc, x_train, y_train)
print "Training data converted into Resilient Distributed Dataset"
spark_model = SparkModel(sc,lstmModel,optimizer=adam ,frequency='epoch', mode='asynchronous', num_workers=2)
print "Spark Model Initialized"
spark_model.train(rddataset, nb_epoch=10, batch_size=200, verbose=1, validation_split=0)  #
print "LSTM model training done !!"
print "Saving weights!!"
spark_model.save_weights("/home/vyassu/my_model_weights.h5")
#print sm.get_server_weights()
print "LSTM model testing commencing !!"
val = np.resize(x_train[count],(1,30,19))
print(y_train[count])
val1= np.resize(x_test[count1-1],(1,30,19))
print(y_test[count1-1])
output=spark_model.predict_classes(val)
output1=spark_model.predict_classes(val1)
print output
print output1
predicted1=spark_model.predict_classes(x_test)
df_confusion = pd.crosstab(y_test.flatten(), predicted1.flatten(), rownames=['Actual'], colnames=['Predicted'], margins=True)
print df_confusion