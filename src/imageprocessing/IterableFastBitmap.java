package imageprocessing;

import Catalano.Imaging.FastBitmap;
import Catalano.Imaging.Filters.GaussianBlur;
import Catalano.Imaging.Filters.HistogramEqualization;
import Catalano.Math.Matrix;
import java.util.Iterator;

/**
 *
 * @author José Carranza
 */
public class IterableFastBitmap extends FastBitmap implements Iterable<double[]>, Iterator<double[]> {

    private int windowWidth, windowHeight, posX, posY;

    private int[][] arrayGray;
    
    private double[][] currentMatrix; 

    private int offsetFactor;
    
    private static int[] mask;
    static {
        FastBitmap temp = new FastBitmap("./data/utils/mask.bmp");
        int[][] temp2 = new int[temp.getHeight()][temp.getWidth()];
        temp.toArrayGray(temp2);
        mask = Matrix.Reshape(temp2);
    }

    public IterableFastBitmap(String pathname, int windowWidth, int windowHeight, int offsetFactor) {
        super(pathname);
        this.windowWidth = windowWidth;
        this.windowHeight = windowHeight;
        this.posX = 0;
        this.posY = 0;
        this.arrayGray = new int[getHeight()][getWidth()];
        this.offsetFactor = offsetFactor;

        try {
            this.toGrayscale();
        } catch (Exception e) {
        }
        this.toArrayGray(arrayGray);
    }

    public int getPosX() {
        return posX;
    }

    public void setOffsetFactor(int offsetFactor) {
        this.offsetFactor = offsetFactor;
    }

    public int getOffsetFactor() {
        return offsetFactor;
    }

    public void setPosX(int posX) {
        this.posX = posX;
    }

    public int getPosY() {
        return posY;
    }

    public void setPosY(int posY) {
        this.posY = posY;
    }

    public int getWindowWidth() {
        return windowWidth;
    }

    public void setWindowWidth(int windowWidth) {
        this.windowWidth = windowWidth;
    }

    public int getWindowHeight() {
        return windowHeight;
    }

    public void setHWindoweight(int windowHeight) {
        this.windowHeight = windowHeight;
    }

    @Override
    public Iterator<double[]> iterator() {
        this.posX = 0;
        this.posY = 0;
        refreshArray();
        return this;
    }

    @Override
    public boolean hasNext() {
        return (this.posX <= getWidth() - windowWidth) && (this.posY <= getHeight() - windowHeight);
    }

    @Override
    public double[] next() {
//        long t = System.currentTimeMillis();
        int[][] res = Matrix.Submatrix(arrayGray, posY, posY + windowHeight - 1, posX, posX + windowWidth - 1);
        posX += offsetFactor;
        if ((posX + windowWidth) >= getWidth()) {
            posX = 0;
            posY += offsetFactor;
        }
        FastBitmap b = new FastBitmap(res);
        HistogramEqualization he = new HistogramEqualization();
        GaussianBlur g = new GaussianBlur(1);
        he.applyInPlace(b);
//        g.applyInPlace(b);
        b.toArrayGray(res);
//        b.saveAsJPG("/Users/josecarranza/Desktop/test/"+posX+"-"+posY+".jpg");
        double[][] r = Matrix.Reshape(Matrix.toDoubleArray(res),windowWidth,windowHeight);
        currentMatrix = Matrix.Copy(r);
        Matrix.Divide(r, 255);
        double[] answer = Matrix.Reshape(r);
//        int[] intans = new int[answer.length];
//        for (int i = 0; i < intans.length; i++) {
//           intans[i] = (int)answer[i]; 
//        }
//        Matrix.Divide(r, 255);
//        answer = Matrix.Reshape(r);
        for (int i = 0; i < answer.length; i++) {
            if(mask[i] == 0){
                answer[i] = 0;
//                intans[i] = 0;
            }
        }
        
//        new FastBitmap(Matrix.Reshape(intans, 19, 19)).saveAsJPG("/Users/josecarranza/Desktop/masked/"+posX+"-"+posY+".jpg");
//        System.out.println("start:"+t);
//        System.out.println("end:"+System.currentTimeMillis());
        return answer;
    }

    public void refreshArray() {
        arrayGray = new int[getHeight()][getWidth()];
        this.toArrayGray(arrayGray);
    }
    
    
    public double[][] getCurrentMatrix(){
        return this.currentMatrix;
    }

}
