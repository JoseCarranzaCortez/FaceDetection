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
     * @param datos Conjunto de datos de los cuales conocemos las clases reales.
     * @param clas Classifier utilizado.
     * @return Tasa de fallo del clasificador.
     */
    public double error(Data datos, Classifier clas) {
        // Clasificamos el conjunto de datos
        ArrayList<Integer> clases = clas.classify(datos);

        if (PRINT_RESULTS) {
            System.out.println("\n\nReal classes and predictions:");
        }
        double falsosPositivos = 0;
        double error = 0;
        for (int i = 0; i < datos.getNumDatos(); i++) {
            // La clase real se encuentra en la última posición del array.
            double a = datos.getDatos().get(i)[datos.getClassIndex()];
//            if (a == 0) {
//                a = -1;
//            }
            int b = clases.get(i);

            if (PRINT_RESULTS) {
                System.out.println("Clase real: " + a + ", Predicción:: " + (double) b);
            }
            if ((int) a != b) {
                // El clasificador no clasificó correctamente.
                if(b == 1){
                    falsosPositivos++;
                }
                error++;
            }
        }
        System.out.println("falsos positivos: "+falsosPositivos+". "+(100d*falsosPositivos/error)+"%");
        // Devolvemos porcentaje de error
        return 100 * error / datos.getNumDatos();
    }

    // Realiza una clasificacion utilizando una estrategia de particionado determinada
    /**
     * Realiza una clasificacion utilizando una estrategia de particionado determinada.
     *
     * @param part Estrategia de particionado.
     * @param datos Conjunto de datos.
     * @param clas Classifier.
     * @return Lista de errores de todas las particiones utilizadas.
     */
    public static ArrayList<Double> validacion(SplittingStrategy part, Data datos, Classifier clas) {
        //Creamos las particiones siguiendo la estrategia llamando a part.creaParticiones
        ArrayList<Partition> p;                         // Lista de particiones 
        ArrayList<Double> errores = new ArrayList<>();  // Lista de errores
        p = part.crearParticiones(datos);

        for (Partition p1 : p) {
            // Entrenamos utilizando el conjunto de training
            clas.training(datos.extraeDatosTrain(p1));
            System.out.println("Entrenamiento terminado");
            // Clasificamos utilizando el conjunto de prueba
            double error = clas.error(datos.extraeDatosTest(p1), clas);

            // Agregamos el porcentaje de error a la lista de errores.
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
            System.out.println("Error guardando el clasificador en: "+path);
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
            System.out.println("Error leyendo el clasificador en: "+path);
            Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }
}
