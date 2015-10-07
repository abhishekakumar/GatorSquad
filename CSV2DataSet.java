package com.depies.dl4j_tutorial;
import java.io.BufferedReader;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.*;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

public class CSV2DataSet {
	INDArray label = Nd4j.create(345677, 1);
	INDArray feature = Nd4j.create(345677, 1);
	DataSet inputDataSet  = null;
	
	public static void main(String[] args) {
		// Creating class variable 
		String csvFile = "C://Users/vyas/Downloads/2013/2013-04-10.csv";
			CSV2DataSet test = new CSV2DataSet();
			test.loadFile(csvFile);
	}
	/*
	 * Function to convert CSV data into DATASET class variable
	 * 
	 */
	public DataSet loadFile(String Filename) {	
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		int count = 0;
		try 
		{
			br = new BufferedReader(new FileReader(Filename));   // Read each line from the input CSV
			// Loop till end of File
			while ((line = br.readLine()) != null) 
			{
			   if (count >0)  // Condition to filter out row1 from the Excel sheet
				{
				   String[] data = line.split(cvsSplitBy);
				   label.putScalar(count,Integer.parseInt(data[4]));
				   feature.putScalar(count,Integer.parseInt(data[6]));
				   //System.out.println("Failure " + data[4]
	                                // + " , Input=" + data[6] + "]");
				}
				count++;
			}
			inputDataSet = new DataSet(label, feature);
			//System.out.println(inputDataSet);
			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return inputDataSet;
	}
}
