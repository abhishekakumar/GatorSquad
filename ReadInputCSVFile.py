from StringIO import StringIO
import pandas as pd 
import glob as gb
import os

#Method to generate Dataframes from CSV file data
def generate_DataFrame(path,SMARTParams):
    #Variable to store Input Data Frame
    input_data_frame = pd.DataFrame()
    #List of DataFrames sorted by Model number
    dict_input_data_frame={}

    SMARTParams=SMARTParams+['model','serial_number','failure']
    print SMARTParams
    # Loop to traverse through the Filesystem path to get the csv files
    for f in gb.glob(path):      #"/home/user/Desktop/Cloud/2013*.csv"
        # Temp dataframe to store values read from each csv file 
        temp_dataframe = pd.read_csv(f,sep=',',usecols=SMARTParams)
        # Appending contents of each csv file to the Input Dataframe
        input_data_frame = input_data_frame.append(temp_dataframe,ignore_index=True,verify_integrity=False)
    # End of Loop
    # Storing the Dataframe sorted according to the various Disk Models
    groupedDataFrame = input_data_frame.groupby(['model'])

    # Appending Failure columns to the list of SMART parameters
    SMARTParams.remove('model')
    SMARTParams.remove('serial_number')

    # Loop to traverse through the Dataframes and storing each subDataframe as a list
    for name,group in groupedDataFrame:
        dict_input_data_frame.update({name:group})
        filename=os.environ.get('MODEL_CSV_FILEPATH')+str(name).replace(' ','_')+'.csv'
        temp_group = pd.DataFrame(data=group,columns= SMARTParams)
        temp_group.to_csv(filename)


if __name__ == '__main__':
     returnedValue = generate_DataFrame("/home/user/Desktop/Cloud/Test/2014/2014-06*.csv")
     #print returnedValue
     
     


