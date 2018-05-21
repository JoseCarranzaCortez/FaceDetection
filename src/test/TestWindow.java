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
        
        Classifier c1 = Classifier.readFromDisk(args[0]);
        Classifier c2 = Classifier.readFromDisk(args[1]);
                
        String image = "facial";
        String extension = ".jpg";
        
        String filePath = "/Users/josecarranza/Desktop/" + image + extension;
        String outputFilePath = "/Users/josecarranza/Desktop/" + image + "_processed" + extension;
       
        NeuralNetworkFaceDetector faceDetector = new NeuralNetworkFaceDetector(c1,c2);
        
        faceDetector.detectFaces(filePath, outputFilePath);
        
    }
}
