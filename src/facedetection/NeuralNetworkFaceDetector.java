package facedetection;

import Catalano.Imaging.FastBitmap;
import Catalano.Imaging.FastGraphics;
import Catalano.Imaging.Filters.ResizeBicubic;
import classifiers.Classifier;
import imageprocessing.IterableFastBitmap;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * @author José Carranza
 */
public class NeuralNetworkFaceDetector {
    
    Classifier c1;
    Classifier c2;
    
    public NeuralNetworkFaceDetector(Classifier c1, Classifier c2){
        this.c1 = c1;
        this.c2 = c2;
    }
    
    
    
    public void detectFaces(String filePath, String outputFilePath, float tol){
        
        ArrayList<Integer> xList = new ArrayList<>();
        ArrayList<Integer> yList = new ArrayList<>();
        ArrayList<Double> points1 = new ArrayList<>();
        ArrayList<Double> points2 = new ArrayList<>();
        ArrayList<Double> points3 = new ArrayList<>();
        ArrayList<Double> scales = new ArrayList<>();

        ResizeBicubic rb;
        
        
        // Open the image using an iterable window
        IterableFastBitmap fb = new IterableFastBitmap(filePath, 19, 19, 1);
        //HistogramStretch he = new HistogramStretch();
        //AdaptiveContrastEnhancement he = new AdaptiveContrastEnhancement(32, .2, .6, 1, 2);
        // Image preprocessing
        //he.applyInPlace(fb);
        // Get a copy of the image in which to draw results
        FastBitmap fbOut = new FastBitmap(filePath);
        FastGraphics fg = new FastGraphics(fbOut);
        
        double scale = 1;
        double nw = 500;
        // Re scale the image to the new width 
        if (fb.getWidth() > nw) {
            scale = nw / fb.getWidth();
            rb = new ResizeBicubic((int) nw, (int) (nw * fb.getHeight() / fb.getWidth()));
            rb.applyInPlace(fb);

            fb.refreshArray();
            //fb.saveAsJPG("./preprocessed.jpg");
        }

        main:
        while (fb.getWidth() >= 19 && fb.getHeight() >= 19) {
            System.out.println("Dimensions:" + fb.getWidth() + " x " + fb.getHeight());
            Iterator<double[]> window = fb.iterator();
            int caras = 0;
            int counter = 0;
            while (window.hasNext()) {
                double[] w = window.next();
                /* w is the portion of the image that is being scanned. 
                 * Any early rejection should be done here.
                 */
                
                /*if(fr.reject(fb.getCurrentMatrix())){
                    continue;
                }*/
                
                double clase1 = c1.getScore(w);
                double clase2 = c2.getScore(w);
                if (clase1 > tol && clase2 > tol) {
                    caras++;
                    System.out.println("Face found on coordinates: [" + fb.getPosX() + ", " + fb.getPosY() + "]-> C1 score: " + clase1 + ", C2 score: "+clase2);
                    xList.add(fb.getPosX());
                    yList.add(fb.getPosY());
                    points1.add(clase1);
                    points2.add(clase2);
                    scales.add(scale);
                }
            }
            System.out.println("Number of faces found:" + caras);
            int newWidth = (int) (fb.getWidth() * 0.8);
            int newHeight = (int) (fb.getHeight() * 0.8);
            scale *= 0.8;
            rb = new ResizeBicubic(newWidth, newHeight);
            rb.applyInPlace(fb);
        }

        for (int i = 0; i < xList.size(); i++) {
            int red = 255 - (int) (255 * points1.get(i));
            int green = (int) (255 * points1.get(i));
            fbOut.toRGB();
            fg.setColor(red, green, 0);
            if (points1.get(i) >= 0.3 && points2.get(i) >= 0.3) {
                int size = (int) (19d / scales.get(i));
                //System.out.println(size);
                int x = (int) ((double) xList.get(i) / scales.get(i));
                int y = (int) ((double) yList.get(i) / scales.get(i));
                if (!((y + size >= fbOut.getHeight()) || (x + size >= fbOut.getWidth()))) {
                    fg.DrawRectangle(y, x, size, size);
                }
            }
        }
        fbOut.saveAsJPG(outputFilePath);
    
    }
}
