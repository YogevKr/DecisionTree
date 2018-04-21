package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex = -1;
	double returnValue = 0;
	double impurity = 1;
	int depth;
}

public class DecisionTree implements Classifier {
	private Node m_rootNode;
	private ImpurityCriterion m_impurityCriterion = ImpurityCriterion.Gini;
	private int m_maxHeight = 0;
	private int m_sumOfHeights = 0;
	private int m_instancesClassified = 0;
	private int m_numOfClassValues = 0;
	private int m_pruningValueIndex = 0;

	private static final double PERFECTLY_CLASSIFIED = 0;
	private static final double[] ALLOWED_P_VALUES = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
	private static final double[][] CHI_SQUARED_DISTRIBUTION =
			{
					{0,0,0,0,0,0,0,0,0,0,0,0},
					{0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 8.071, 5.899, 6.737, 7.584, 8.438},
					{0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340},
					{1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845},
					{3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026},
					{7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300},
			};

	@Override
	public void buildClassifier(Instances arg0) throws Exception
	{
		m_numOfClassValues = arg0.numClasses();
		m_rootNode = new Node();
		buildTree(m_rootNode, arg0);
	}

	@Override
	public double classifyInstance(Instance instance)
	{
		Node currentNode = m_rootNode;
		int currentDepth = 0;
		while (!isLeaf(currentNode))
		{
			int nodeAttributeIndex = currentNode.attributeIndex;
			int instanceAttributeValueIndex = (int) instance.value(nodeAttributeIndex);
			Node nextNode = currentNode.children[instanceAttributeValueIndex];
			if (nextNode == null) break;
			currentNode = nextNode;
			currentDepth++;
		}

		if (currentDepth > m_maxHeight)
		{
			m_maxHeight = currentDepth;
		}

		currentNode.depth = currentDepth;
		m_instancesClassified++;
		m_sumOfHeights += currentDepth;

		return currentNode.returnValue;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception
	{
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities()
	{
		// Don't change
		return null;
	}

	private void buildTree(Node node, Instances instances) throws Exception
	{
		double[] probabilities = getProbabilitiesForInstances(instances);
		node.returnValue = calcReturnValue(probabilities);

		node.impurity = calcImpurity(probabilities);
		if (node.impurity == PERFECTLY_CLASSIFIED) return;

		node.attributeIndex = findBestAttribute(instances);
		if (node.attributeIndex == -1) return;

		if ((m_pruningValueIndex != 0) && (pruningCheck(instances, node.attributeIndex))) return;

		int numOfChildren = instances.attribute(node.attributeIndex).numValues();
		Node[] childNodes = new Node[numOfChildren];
		Instances[] filteredInstances = getFilteredInstances(instances, node.attributeIndex);
		for (int valueIndex = 0; valueIndex < numOfChildren; valueIndex++)
		{
			if (filteredInstances[valueIndex].numInstances() == 0)
			{
				childNodes[valueIndex] = null;
				continue;
			}
			childNodes[valueIndex] = new Node();
			childNodes[valueIndex].parent = node;
			buildTree(childNodes[valueIndex], filteredInstances[valueIndex]);
		}

		node.children = childNodes;


	}

	private Instances[] getFilteredInstances(Instances instances, int attributeIndex) throws Exception
	{
		int numOfValues = instances.attribute(attributeIndex).numValues();
		Instances[] filteredInstances = new Instances[numOfValues];

		for (int valueIndex = 0; valueIndex < numOfValues; valueIndex++)
		{
			RemoveWithValues remove = new RemoveWithValues();
			remove.setNominalIndices(valueIndex + 1 + "");
			remove.setAttributeIndex(attributeIndex + 1 + "");
			remove.setInvertSelection(true);
			remove.setInputFormat(instances);

			filteredInstances[valueIndex] = Filter.useFilter(instances, remove);
		}

		return filteredInstances;
	}

	private int findBestAttribute(Instances instances)
	{
		double bestGain = 0;
		int bestIndex = 0;

		for (int i = 0; i < instances.numAttributes() - 1; i++)
		{
			double currentGain = calcGain(instances, i);
			if (currentGain > bestGain)
			{
				bestGain = currentGain;
				bestIndex = i;
			}
		}

		if (bestGain == 0)
		{
			return -1;
		}

		return bestIndex;
	}

	private double calcGain(Instances instances, int attributeIndex)
	{
		double goodnessOfSplit = calcImpurity(getProbabilitiesForInstances(instances));
		double[] allValueProbabilities = getProbabilitiesByAttribute(instances, attributeIndex);
		int numOfValues = instances.attribute(attributeIndex).numValues();

		for (int i = 0; i < numOfValues; i++)
		{
			double[] specificValueProbabilities = getProbabilitiesByAttributeAndValue(instances, attributeIndex, i);
			double gain = calcImpurity(specificValueProbabilities);
			gain *= allValueProbabilities[i];

			goodnessOfSplit -= gain;
		}

		return goodnessOfSplit;
	}

	public double CalcAvgError(Instances validationSet)
	{
		m_maxHeight = 0;
		m_sumOfHeights = 0;
		m_instancesClassified = 0;

		double numOfWrongClassifications = 0;

		for (int i = 0; i < validationSet.numInstances(); i++)
		{
			Instance currentInstance = validationSet.get(i);
			double currentInstanceClassification = classifyInstance(currentInstance);
			if (currentInstance.classValue() != currentInstanceClassification)
			{
				numOfWrongClassifications++;
			}
		}

		return (numOfWrongClassifications / validationSet.numInstances());
	}

	private double calcImpurity(double[] probabilities)
	{
		double impurity;

		if (m_impurityCriterion == ImpurityCriterion.Gini)
		{
			impurity = calcGiniIndex(probabilities);
		} else
		{
			impurity = calcEntropy(probabilities);
		}

		return impurity;
	}

	private double calcEntropy(double[] probabilities)
	{
		double entropyValue = 0;

		for (int i = 0; i < probabilities.length; i++)
		{
			if(probabilities[i] == 0) continue;

			entropyValue += probabilities[i] * Math.log(probabilities[i]) / Math.log(2);
		}

		return -entropyValue;
	}

	private double calcGiniIndex(double[] probabilities)
	{
		double probabilitiesSquaredSum = 0;
		for (int i = 0; i < probabilities.length; i++)
		{
			probabilitiesSquaredSum += Math.pow(probabilities[i], 2);
		}

		return (1 - probabilitiesSquaredSum);
	}

	private boolean pruningCheck(Instances instances, int attributeIndex)
	{
		int numValues = instances.numDistinctValues(attributeIndex);
		int freedomDegree = numValues - 1;
		double chiSquare = calcChiSquare(instances, attributeIndex);

		double nodePruningValue = CHI_SQUARED_DISTRIBUTION[m_pruningValueIndex][freedomDegree - 1];

		return nodePruningValue > chiSquare;

	}

	private double calcChiSquare(Instances instances, int attributeXj)
	{
		double chiSquare = 0;
		int numOfBrackets = instances.attribute(attributeXj).numValues();

		int[] D = new int[numOfBrackets]; // Df
		int[] p = new int[numOfBrackets]; // pf
		int[] n = new int[numOfBrackets]; // nf
		double[] E0 = new double[numOfBrackets];
		double[] E1 = new double[numOfBrackets];
		double[] P = new double[2]; // P(Y = 0), P(Y = 1)

		for (int i = 0; i < instances.numInstances(); i++)
		{
			Instance currentInstance = instances.get(i);

			// P(Y = 0), P(Y = 1)
			int classValueIndex = (int) currentInstance.classValue();
			P[classValueIndex]++;

			// Df
			int attributeValueIndex = (int) currentInstance.value(attributeXj);
			D[attributeValueIndex]++;

			// pf, nf
			if (classValueIndex == 0)
			{
				p[attributeValueIndex]++;
			}
			else {
				n[attributeValueIndex]++;
			}

		}

		P[0] /= instances.numInstances();
		P[1] /= instances.numInstances();

		// E0, E1
		for (int i = 0; i < numOfBrackets; i++)
		{
			E0[i] = D[i] * P[0];
			E1[i] = D[i] * P[1];
		}

		for (int i = 0; i < numOfBrackets; i++)
		{
			if(D[i] == 0) continue;
			chiSquare += ((((p[i] - E0[i]) * (p[i] - E0[i])) / E0[i])) + ((((n[i] - E1[i]) * (n[i] - E1[i])) / E1[i]));
		}

		return chiSquare;
	}

	private double calcReturnValue(double[] probabilities)
	{
		double bestGuess = 0;
		int returnValue = 0;

		for (int i = 0; i < probabilities.length; i++)
		{
			if (probabilities[i] > bestGuess)
			{
				bestGuess = probabilities[i];
				returnValue = i;
			}
		}

		return returnValue;
	}

	private double[] getProbabilitiesForInstances(Instances instances)
	{
		double[] probabilities = new double[m_numOfClassValues];

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance currentInstance = instances.get(i);
			int classValue = (int) currentInstance.classValue();
			probabilities[classValue]++;
		}

		for (int i =0; i < probabilities.length; i++) {
			probabilities[i] = probabilities[i] / instances.numInstances();
		}

		return probabilities;
	}

	private double[] getProbabilitiesByAttribute(Instances instances, int attributeIndex)
	{
		double[] probabilities = new double[instances.attribute(attributeIndex).numValues()];

		for (int i = 0; i < instances.numInstances(); i++)
		{
			Instance currentInstance = instances.get(i);
			int attributeValueIndex = (int) currentInstance.value(attributeIndex);
			probabilities[attributeValueIndex]++;
		}

		for (int i = 0; i < probabilities.length; i++)
		{
			probabilities[i] = probabilities[i] / instances.numInstances();
		}
		return probabilities;
	}

	private double[] getProbabilitiesByAttributeAndValue(Instances instances, int attributeIndex, int attributeValue)
	{
		double[] probabilities = new double[m_numOfClassValues];

		int totalRelevantInstances = 0;

		for (int i = 0; i < instances.numInstances(); i++)
		{
			Instance currentInstance = instances.get(i);
			int currentInstanceValue = (int) currentInstance.value(attributeIndex);
			if (currentInstanceValue == attributeValue)
			{
				int classValueIndex = (int) currentInstance.classValue();
				probabilities[classValueIndex]++;
				totalRelevantInstances++;
			}
		}

		if (totalRelevantInstances != 0)
		{
			for (int i = 0; i < probabilities.length; i++)
			{
				probabilities[i] = probabilities[i] / totalRelevantInstances;
			}
		}

		return probabilities;
	}

	public void SetImpurityMeasure(ImpurityCriterion chosenCriterion)
	{
		m_impurityCriterion = chosenCriterion;
	}

	public void SetPruningValue(double value)
	{
		for (int i = 0; i < ALLOWED_P_VALUES.length; i++)
		{
			if (value == ALLOWED_P_VALUES[i])
			{
				m_pruningValueIndex = i;
			}
		}
	}

	public void printNode()
	{
		printNode(m_rootNode, 0);
	}

	private void printNode(Node S, int numOfTabs)
	{
		if (S != null)
		{
			String tabs = printSomeTabs(numOfTabs);

			if (S.parent == null)
			{
				System.out.println("Root");
			}

			if (S.children == null)
			{
				System.out.print(tabs + "Leaf. ");
				System.out.println("Retuning value: " + S.returnValue);
			} else
			{
				System.out.println( tabs + "Retuning value: " + S.returnValue);
				for (int i = 0; i < S.children.length; i++)
				{
					if (S.children[i] != null)
					{
						System.out.println(tabs + "If attribute " + S.attributeIndex + " = " + i);
						printNode(S.children[i], numOfTabs + 1);
					}
				}
			}
		}
	}

	private String printSomeTabs(int numOfTabs)
	{
		StringBuilder tabsString = new StringBuilder();
		for (int i = 0; i < numOfTabs; i++)
		{
			tabsString.append("\t");
		}

		return tabsString.toString();
	}

	public double GetAvgHeight()
	{
		return ((double) m_sumOfHeights / m_instancesClassified);
	}

	public int GetMaxHeight()
	{
		return m_maxHeight;
	}

	public static double[] AllowedPValues()
	{
		return ALLOWED_P_VALUES;
	}

	public double GetPruningValue()
	{
		return ALLOWED_P_VALUES[m_pruningValueIndex];
	}

	private boolean isLeaf(Node i_node)
	{
		return (i_node.children == null);
	}

}
