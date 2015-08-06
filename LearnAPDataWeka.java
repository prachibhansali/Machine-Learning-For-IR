import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.*;
import java.util.*;

public class LearnAPDataWeka {

	public static void main(String args[]) throws Exception
	{
		String trainfile = "/Users/prachibhansali/Documents/IR/Assignment1/PseudoOutputs/temp/TrainingData";
		String testfile = "/Users/prachibhansali/Documents/IR/Assignment1/PseudoOutputs/temp/TestData";

		createArffFile(trainfile);
		createArffFile(testfile);
		eval(trainfile+".arff",testfile+".arff");
	}

	private static void eval(String trainfile, String testfile) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(trainfile));
		Instances train = new Instances(reader);
		train.setClassIndex(train.numAttributes() - 1);
		reader.close();

		reader = new BufferedReader(new FileReader(trainfile));
		Instances test = new Instances(reader);
		reader.close();
		
		Classifier cls = new J48();
		cls.buildClassifier(train);

		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(train);
		test.setClassIndex(test.numAttributes() - 1);

		double res [] = eval.evaluateModel(cls, train);

		System.out.println(eval.toSummaryString("\nResults\n======\n", false));

		//PrintWriter pw = new PrintWriter("/Users/prachibhansali/Documents/IR/Assignment6/TestingDataEval");
		PrintWriter pw = new PrintWriter("/Users/prachibhansali/Documents/IR/Assignment6/TrainingDataEval");
		BufferedReader br = new BufferedReader(new FileReader(trainfile.substring(0,trainfile.indexOf("."))));
		String line="";
		int i=0;
		while((line=br.readLine())!=null)
		{
			String str[] = line.split("\\s+");
			if(res[i++]==1)
				pw.println(str[0]+" "+"Q0"+" "+str[1]+" "+"1"+" "+1+" "+"Exp");
			else pw.println(str[0]+" "+"Q0"+" "+str[1]+" "+"1"+" "+0+" "+"Exp");
		}

		pw.close();
		br.close();

	}

	private static void createArffFile(String file) throws Exception{
		PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file+".arff")));
		pw.println("@RELATION docquery");
		pw.println("@ATTRIBUTE count NUMERIC");
		pw.println("@ATTRIBUTE okapi NUMERIC");
		pw.println("@ATTRIBUTE bm NUMERIC");
		pw.println("@ATTRIBUTE tfidf NUMERIC");
		pw.println("@ATTRIBUTE laplace NUMERIC");
		pw.println("@ATTRIBUTE jm NUMERIC");
		//pw.println("@ATTRIBUTE doclength NUMERIC");
		pw.println("@ATTRIBUTE class {0,1}");
		pw.println("@DATA");
		pw.println();

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line="";
		while((line=br.readLine())!=null)
		{
			String[] str = line.split("\t");
			pw.print(str[3]+",");
			pw.print(str[4]+",");
			pw.print(str[5]+",");
			pw.print(str[6]+",");
			pw.print(str[7]+",");
			pw.print(str[8]+",");
			//pw.print(str[9]+",");
			pw.print(str[2]+"\n");
		}
		br.close();
		pw.close();
	}
}