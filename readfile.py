from StringIO import StringIO
import pandas as pd 
import glob as gb
    
def generate_DataFrame(path):
    #Variable to store Input Data Frame
    input_data_frame = pd.DataFrame()
    #List of DataFrames sorted by Model number
    dict_input_data_frame={}
    
    # Loop to traverse through the Filesystem path to get the csv files
    for f in gb.glob(path):      #"/home/user/Desktop/Cloud/2013*.csv"
        # Temp dataframe to store values read from each csv file 
        temp_dataframe = pd.read_csv(f,sep=',',usecols=['model','serial_number','smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_9_raw','smart_10_raw','smart_12_raw',
 'smart_184_raw','smart_187_raw','smart_188_raw','smart_189_raw','smart_190_raw','smart_197_raw','smart_198_raw',
'smart_199_raw','smart_240_raw','smart_241_raw','smart_242_raw','failure'])
        
        # Appending contents of each csv file to the Input Dataframe
        input_data_frame = input_data_frame.append(temp_dataframe,ignore_index=True,verify_integrity=False)
    # End of Loop
    # Storing the Dataframe sorted according to the various Disk Models
    groupedDataFrame = input_data_frame.groupby(['model'])
    
    # Loop to traverse through the Dataframes and storing each subDataframe as a list
    for name,group in groupedDataFrame:
        dict_input_data_frame.update({name:group})
        filename='/hadoop/elephas/Output/'+str(name).replace(' ','_')+'.csv'
        temp_group = pd.DataFrame(data=group,columns=['smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw','smart_7_raw','smart_9_raw','smart_10_raw','smart_12_raw',
 'smart_184_raw','smart_187_raw','smart_188_raw','smart_189_raw','smart_190_raw','smart_197_raw','smart_198_raw',
'smart_199_raw','smart_240_raw','smart_241_raw','smart_242_raw','failure'])
        temp_group.to_csv(filename)

    #print(list_val[1])  #Printing test data
    return  dict_input_data_frame

if __name__ == '__main__':
     returnedValue = generate_DataFrame("/home/user/Desktop/Cloud/Test/2014/2014-06*.csv")
     #print returnedValue
     
     


