/**
 * @author José Carranza
 */
package classifiers;

import data.Data;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import splitting.Partition;
import splitting.SplittingStrategy;

abstract public class Classifier implements Serializable{

    private static final boolean PRINT_RESULTS = false;


    /**
     * Does the training of the classifier using a set of training data.
     * 
     * @param trainData Training Data Set
     */
    abstract public void training(Data trainData);

    /**
     * Classifies a set of data after having made the training.
     *
     * @param testData The set of data that needs to be classified.
     * @return List of classes that have been assigned to the elements.
     */
    abstract public ArrayList<Integer> classify(Data testData);
    
    /**
     * Classify a pattern
     *
     * @param d The pattern to classify in an array form.
     * @return Class
     */
    abstract public int classifyPattern(double[] d);
    
    /**
     * Classifies a pattern.
     *
     * @param d
     * @return Class
     */
    abstract public double getScore(double[] d);

    /**
     * Calculates the error rate of the classifiers.
     *
     * @param dataSet Set of data of which we know real classes.
     * @param clas Classifier used.
     * @return Failure rate of the classifier.
     */
    public double error(Data dataSet, Classifier clas) {
        // First, we classify the dataset
        ArrayList<Integer> clases = clas.classify(dataSet);

        if (PRINT_RESULTS) {
            System.out.println("\n\nReal classes and predictions:");
        }
        double falsePositives = 0;
        double error = 0;
        for (int i = 0; i < dataSet.getDataCount(); i++) {
            // La clase real se encuentra en la última posición del array.
            double a = dataSet.getData().get(i)[dataSet.getClassIndex()];
            int b = clases.get(i);

            if (PRINT_RESULTS) {
                System.out.println("Real Class: " + a + ", Preducted Class: " + (double) b);
            }
            if ((int) a != b) {
                // The classifier did not predict correctly
                if(b == 1){
                    falsePositives++;
                }
                error++;
            }
        }
        System.out.println("Flase positives: "+falsePositives+". "+(100d*falsePositives/error)+"%");
        // Return the error rate
        return 100 * error / dataSet.getDataCount();
    }

    /**
     * Classifies a dataset using a specified splitting strategy
     *
     * @param splittingStrategy Splitting strategy.
     * @param dataSet Dataset to be classified.
     * @param clas Classifier.
     * @return list of errors of all partitions.
     */
    public static ArrayList<Double> validacion(SplittingStrategy splittingStrategy, Data dataSet, Classifier clas) {
        // First, we create the partitions.
        ArrayList<Partition> p;                         // Lista de particiones 
        ArrayList<Double> errores = new ArrayList<>();  // Lista de errores
        p = splittingStrategy.crearParticiones(dataSet);

        for (Partition p1 : p) {
            // Train using current partition's train set.
            clas.training(dataSet.extractTrainData(p1));
            System.out.println("Training Finished");
            // Classigy using the test set.
            double error = clas.error(dataSet.extractTestData(p1), clas);

            // Add error rate to the list
            errores.add(error);
        }
        return errores;
    }

    public void saveToDisk(String path) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(path)));
            oos.writeObject(this);
            oos.close();
        } catch (IOException ex) {
            System.out.println("Error saving classifier on: "+path);
            Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static Classifier readFromDisk(String path) {
       try {
            ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)));
            Classifier c = (Classifier)ois.readObject();
            ois.close();
            return c;
        } catch (Exception ex) {
            System.out.println("Error reading classifier from: "+path);
            Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }
}
