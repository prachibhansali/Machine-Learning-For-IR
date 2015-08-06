import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

import java.io.*;
import java.util.*;

public class LearnAPData {

	static HashMap<String,Boolean> trainingset = new HashMap<String,Boolean>();
	HashMap<String,Boolean> testset = new HashMap<String,Boolean>();

	public static void main(String args[]) throws Exception
	{
		int NUM = 20;
		int FEATURES = 8;
		int TNUM=5;
		Problem problem = new Problem();
		problem.l = NUM;
		problem.n = FEATURES;
		ArrayList<Double> target = new ArrayList<Double>();

		BufferedReader br = new BufferedReader(new FileReader("/Users/prachibhansali/Documents/IR/Assignment1/PseudoOutputs/temp/TrainingData"));
		String line="";
		ArrayList<ArrayList<Feature>> features = new ArrayList<ArrayList<Feature>>();
		while((line=br.readLine())!=null)
		{
			String[] str=line.split("\t");

			trainingset.put(str[0], true);
			if(trainingset.size() > NUM)
				break;

			target.add(Double.parseDouble(str[2]));
			ArrayList<Feature> f = new ArrayList<Feature>();
			if(Integer.parseInt(str[3])!=0) f.add(new FeatureNode(1,Integer.parseInt(str[3])));
			if(Double.parseDouble(str[4])!=0.0) f.add(new FeatureNode(2,Double.parseDouble(str[4])));
			if(Double.parseDouble(str[5])!=0.0) f.add(new FeatureNode(3,Double.parseDouble(str[5])));
			if(Double.parseDouble(str[6])!=0.0) f.add(new FeatureNode(4,Double.parseDouble(str[6])));
			f.add(new FeatureNode(5,Double.parseDouble(str[7])));
			f.add(new FeatureNode(6,Double.parseDouble(str[8])));
			f.add(new FeatureNode(7,Double.parseDouble(str[9])));
			if(Double.parseDouble(str[10])!=0) f.add(new FeatureNode(8,Double.parseDouble(str[10])));

			features.add(f);
		}
		br.close();

		Feature[][] array = new Feature[features.size()][];
		for (int i = 0; i < features.size(); i++) {
			ArrayList<Feature> row = features.get(i);
			array[i] = row.toArray(new Feature[row.size()]);
		}
		
		System.out.println(features.size());
		
		problem.x = array;
		double[]  t = convertToArray(target);
		problem.y = t;

		SolverType solver = SolverType.L2R_LR; // -s 0
		double C = 0.5;    // cost of constraints violation
		double eps = 0.01; 

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);

		//br = new BufferedReader(new FileReader("/Users/prachibhansali/Documents/IR/Assignment1/PseudoOutputs/temp/TrainingData"));
		br = new BufferedReader(new FileReader("/Users/prachibhansali/Documents/IR/Assignment1/PseudoOutputs/temp/TestData"));
		line="";
		features = new ArrayList<ArrayList<Feature>>();
		//PrintWriter pw = new PrintWriter("/Users/prachibhansali/Documents/IR/Assignment5/TrecEval/TrainingPrecisions");
		PrintWriter pw = new PrintWriter("/Users/prachibhansali/Documents/IR/Assignment5/TrecEval/TestPrecisions");
		Set<String> oneScore = new HashSet<String>();
		
		while((line=br.readLine())!=null)
		{
			String[] str=line.split("\t");

			ArrayList<Feature> f = new ArrayList<Feature>();
			if(Integer.parseInt(str[3])!=0) f.add(new FeatureNode(1,Integer.parseInt(str[3])));
			if(Double.parseDouble(str[4])!=0.0) f.add(new FeatureNode(2,Double.parseDouble(str[4])));
			if(Double.parseDouble(str[5])!=0.0) f.add(new FeatureNode(3,Double.parseDouble(str[5])));
			if(Double.parseDouble(str[6])!=0.0) f.add(new FeatureNode(4,Double.parseDouble(str[6])));
			f.add(new FeatureNode(5,Double.parseDouble(str[7])));
			f.add(new FeatureNode(6,Double.parseDouble(str[8])));
			f.add(new FeatureNode(7,Double.parseDouble(str[9])));
			if(Double.parseDouble(str[10])!=0) f.add(new FeatureNode(8,Double.parseDouble(str[10])));

			features.add(f);
			Feature[] instance = convertToFeatureArray(f);
			double prediction = Linear.predict(model, instance);
			System.out.println(str[0]+" "+str[1]+" "+str[2]+" "+prediction);
			if(((int)(prediction))==1) 
				oneScore.add(str[0]+"-"+str[1]);

		}
		/*int rank85=1;
		int rank56=1;
		int rank59=1;
		int rank71=1;
		int rank64=1;
		*/
		System.out.println(oneScore.size());

		for(String s : oneScore)
		{
			String[] str = s.split("-");
			pw.println(str[0]+" "+"Q0"+" "+str[1]+"-"+str[2]+" "+ 1 +" "+1+" "+"Exp");
		}
		
		br.close();
		pw.close();
	}

	private static Feature[] convertToFeatureArray(ArrayList<Feature> f) {
		Feature t[] = new Feature[f.size()];
		int i=0;
		for(Feature d : f)
			t[i++]=d;
		return t;
	}

	private static double[] convertToArray(ArrayList<Double> target) {
		double t[] = new double[target.size()];
		int i=0;
		for(double d : target)
			t[i++]=d;
		return t;
	}



}
