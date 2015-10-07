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
	INDArray test = Nd4j.create(345677, 1);
	INDArray test1 = Nd4j.create(345677, 1);
	DataSet inputDataSet  = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
			CSV2DataSet test = new CSV2DataSet();
			test.loadFile();
	}
	
	public void loadFile() {

		String csvFile = "C://Users/vyas/Downloads/2013/2013-04-10.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		int count = 0;
		try {
			br = new BufferedReader(new FileReader(csvFile));
			while ((line = br.readLine()) != null) {

			        // use comma as separator
				if (count >0)
				{String[] data = line.split(cvsSplitBy);
				
                test.putScalar(count,Integer.parseInt(data[4]));
                test1.putScalar(count,Integer.parseInt(data[6]));
				System.out.println("Failure " + data[4]
	                                 + " , Input=" + data[6] + "]");
				}
				count++;

			}
			inputDataSet = new DataSet(test, test1);
			System.out.println(inputDataSet);

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
	}
}
