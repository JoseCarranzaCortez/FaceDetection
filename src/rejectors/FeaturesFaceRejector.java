package rejectors;

import Catalano.Imaging.FastBitmap;
import Catalano.Math.Matrix;
import Catalano.Statistics.DescriptiveStatistics;

/**
 * This rejector rejects a face based on facial features. If the rejector does not 
 * see evidence of facial features like eyes, nose and mouth, it will determine that the
 * image is not a face. 
 * @author José Carranza
 */
public class FeaturesFaceRejector implements FaceRejector {
    
    
    private static final double E_D_TRESHOLD = 30000;
    private static final double K = 1.3;
    
    @Override
    public boolean reject(double[][] image) {
        
        // First, we will get each of the sections. 
        // Section A
        double[][] sectionA = Matrix.Submatrix(image, 4, 6, 0, 6);
        
        //FastBitmap fb = new FastBitmap(7, 3, FastBitmap.ColorSpace.Grayscale);
        //fb.arrayToImage(sectionA);
        //fb.saveAsJPG("/Users/josecarranza/Desktop/sectionA.jpg");
        
        
        double[][] sectionB = Matrix.Submatrix(image, 4, 6, 7, 11);
        
        //fb = new FastBitmap(5, 3, FastBitmap.ColorSpace.Grayscale);
        //fb.arrayToImage(sectionB);
        //fb.saveAsJPG("/Users/josecarranza/Desktop/sectionB.jpg");
        
        double[][] sectionC = Matrix.Submatrix(image, 4, 6, 12, 18);
        
        //fb = new FastBitmap(7, 3, FastBitmap.ColorSpace.Grayscale);
        //fb.arrayToImage(sectionC);
        //fb.saveAsJPG("/Users/josecarranza/Desktop/sectionC.jpg");
        
        double[][] sectionD = Matrix.Submatrix(image, 9, 13, 0, 18);
        
        //fb = new FastBitmap(19, 5, FastBitmap.ColorSpace.Grayscale);
        //fb.arrayToImage(sectionD);
        //fb.saveAsJPG("/Users/josecarranza/Desktop/sectionD.jpg");
        
        double[][] sectionE = Matrix.Submatrix(image, 14, 18, 0, 18);
        
        //fb = new FastBitmap(19, 5, FastBitmap.ColorSpace.Grayscale);
        //fb.arrayToImage(sectionE);
        //fb.saveAsJPG("/Users/josecarranza/Desktop/sectionE.jpg");
        
        double[] sectionAArray = Matrix.toDoubleArray(sectionA);
        double[] sectionCArray = Matrix.toDoubleArray(sectionC);
        double[] sectionDArray = Matrix.toDoubleArray(sectionD);
        double[] sectionEArray = Matrix.toDoubleArray(sectionE);
        
        double dVariance = DescriptiveStatistics.Variance(sectionDArray);
        double eVariance = DescriptiveStatistics.Variance(sectionEArray);
        
        
        //System.out.println("E variance: "+eVariance);
        //System.out.println("D variance: "+dVariance);
        
        if(eVariance < E_D_TRESHOLD || dVariance < E_D_TRESHOLD){
            return true;
        }
        
        
        double meanA = DescriptiveStatistics.Mean(sectionAArray);
        double meanC = DescriptiveStatistics.Mean(sectionCArray);
        
        double averageAAboveTreshold = 0;
        double averageCAboveTreshold = 0;
        
        int count = 0;
        
        for(double val:sectionDArray){
            if(val > meanA){
                averageAAboveTreshold += val;
                count++;
            }
        }
        
        averageAAboveTreshold /= count;
        count = 0;
        
        for(double val:sectionEArray){
            if(val > meanC){
                averageCAboveTreshold += val;
                count++;
            }
        }
        averageCAboveTreshold /= count;
        
        //System.out.println("Mean A Above: " + averageAAboveTreshold);
        //System.out.println("Mean C Above: " + averageCAboveTreshold);
        
        if(averageAAboveTreshold < K * averageCAboveTreshold || averageCAboveTreshold < K * averageAAboveTreshold){
            return true;
        }
        
        return false;
    }

}
