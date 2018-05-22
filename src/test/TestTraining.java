package test;

import classifiers.MultilayerPerceptron;
import classifiers.Classifier;
import data.Data;
import java.io.IOException;
import java.util.ArrayList;
import splitting.SimpleValidation;
import splitting.SplittingStrategy;

/**
 *
 * @author José Carranza
 */
public class TestTraining {

    public static void main(String[] args) throws IOException {
        Data d = Data.loadFile(args[1], true);
        
        // Creamos una estrategia de particionado
        SplittingStrategy part = new SimpleValidation();

        int neuronasEntrada = d.getAttributeCount();
        int neuronasOcultas = Integer.parseInt(args[3]);
        int neuronasSalida = d.getClassCount();

        // Creamos un clasificador
        // 100-8%
        Classifier c = new MultilayerPerceptron(neuronasEntrada, neuronasOcultas, neuronasSalida, 0.03, 1,"/Users/josecarranza/Desktop/ecm.csv",200);

        // Obtenemos la tasa de fallo del clasificador
        ArrayList<Double> errores = Classifier.validacion(part, d, c); // Se imprimen*/

        System.out.println("Errores:"+errores);
        
        d = Data.loadFile("./data/test/test.txt",true);
        
        double error = c.error(d, c);
        
        System.out.println("Error para test con "+d.getDataCount()+" datos : "+ error);
        
        c.saveToDisk("/Users/josecarranza/Desktop/faces.perceptron");
    }
}
