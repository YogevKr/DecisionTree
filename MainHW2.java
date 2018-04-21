package HomeWork2;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

enum ImpurityCriterion { Gini, Entropy }

public class MainHW2
{

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {

		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

		// Build an Entropy tree and calculate its validation error
		DecisionTree entropyTree = new DecisionTree();
		entropyTree.SetImpurityMeasure(ImpurityCriterion.Entropy);
		entropyTree.SetPruningValue(1.0);
		entropyTree.buildClassifier(trainingCancer);
		double entropyValidationError = entropyTree.CalcAvgError(validationCancer);

		// Build an Entropy tree and calculate its validation error
		DecisionTree giniTree = new DecisionTree();
		giniTree.SetImpurityMeasure(ImpurityCriterion.Gini);
		giniTree.SetPruningValue(1.0);
		giniTree.buildClassifier(trainingCancer);
		double giniValidationError = giniTree.CalcAvgError(validationCancer);

		ImpurityCriterion chosenCriterion;

		if (entropyValidationError < giniValidationError)
		{
			chosenCriterion = (ImpurityCriterion.Entropy);
		} else
		{
			chosenCriterion = (ImpurityCriterion.Gini);
		}

		System.out.println("Validation error using Entropy: " + entropyValidationError);
		System.out.println("Validation error using Gini: " + giniValidationError);
		System.out.println("----------------------------------------------------");

		DecisionTree bestTree = entropyTree;
		double bestPValue = -1;
		double bestError = 1;
		for (int i = 0; i < DecisionTree.AllowedPValues().length; i++)
		{
			DecisionTree newTree = new DecisionTree();
			newTree.SetImpurityMeasure(chosenCriterion);
			newTree.SetPruningValue(DecisionTree.AllowedPValues()[i]);
			newTree.buildClassifier(trainingCancer);
			double trainingError = newTree.CalcAvgError(trainingCancer);
			double validationError = newTree.CalcAvgError(validationCancer);

			if (validationError < bestError)
			{
				bestTree = newTree;
				bestPValue = newTree.GetPruningValue();
				bestError = validationError;
			}

			System.out.println("Decision Tree with p_value of: " + newTree.GetPruningValue());
			System.out.println("The train error of the decision tree is: " + trainingError);
			System.out.println("Max height on validation data: " + newTree.GetMaxHeight());
			System.out.println("Average height on validation data: " + newTree.GetAvgHeight());
			System.out.println("The validation error of the decision tree is: " + validationError);
			System.out.println("----------------------------------------------------");
		}

		double bestTreeTestingError = bestTree.CalcAvgError(testingCancer);

		System.out.println("Best validation error at p_value = " + bestPValue);
		System.out.println("Test error with best tree: " + bestTreeTestingError);
		System.out.println("Representation of the best tree by ‘if statements’:");
		bestTree.printNode();

	}
}
