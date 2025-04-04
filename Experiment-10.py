import java.util.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.classifiers.Evaluation;

public class NaiveBayesTextClassifier {
    public static void main(String[] args) throws Exception {
        // Create dataset attributes
        FastVector attributes = new FastVector(2);
        attributes.addElement(new Attribute("text", (FastVector) null)); // Text attribute
        attributes.addElement(new Attribute("class")); // Class label

        // Create training data
        Instances trainData = new Instances("TrainInstances", attributes, 10);
        trainData.setClassIndex(1);

        // Add training instances
        trainData.add(new Instance(1.0, new double[]{trainData.attribute(0).addStringValue("spam message"), 0}));
        trainData.add(new Instance(1.0, new double[]{trainData.attribute(0).addStringValue("important meeting"), 1}));
        trainData.add(new Instance(1.0, new double[]{trainData.attribute(0).addStringValue("buy now"), 0}));
        trainData.add(new Instance(1.0, new double[]{trainData.attribute(0).addStringValue("project deadline"), 1}));

        // Train Naive Bayes model
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainData);

        // Evaluate model
        Evaluation eval = new Evaluation(trainData);
        eval.crossValidateModel(nb, trainData, 10, new Random(1));

        // Print accuracy, precision, recall
        System.out.println("Accuracy: " + (1 - eval.errorRate()) * 100 + "%");
        System.out.println("Precision: " + eval.precision(1));
        System.out.println("Recall: " + eval.recall(1));
    }
}
