package test;

import classifiers.Classifier;
import facedetection.NeuralNetworkFaceDetector;
import java.io.IOException;
/**
 *
 * @author José Carranza
 */
public class TestWindow {

    public static void main(String[] args) throws IOException {
        
        if(args.length < 10){
            System.out.println("The number of parameters is not correct.");
            return;
        }
        
        Classifier c1 = Classifier.readFromDisk(args[1]);
        Classifier c2 = Classifier.readFromDisk(args[3]);
        
        String filePath = args[5];
        String outputFilePath = args[7];
        
        float tol = 0.0f;
        try{
            tol = Float.parseFloat(args[9]);
            if(tol < 0 || tol > 1){
                throw new NumberFormatException();
            }
        } catch(NumberFormatException nfe){
            System.out.println("The tolerance parameter should be a floating point number between 0 and 1.");
        }
        
        System.out.println(filePath);
        System.out.println(outputFilePath);
       
        NeuralNetworkFaceDetector faceDetector = new NeuralNetworkFaceDetector(c1, c2);
        
        faceDetector.detectFaces(filePath, outputFilePath, tol);
        
    }
}
