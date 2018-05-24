package classifiers;

import classifiers.Classifier;
import data.Data;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author José Carranza
 */
public class MultilayerPerceptron extends Classifier implements Serializable{

    /*  Activation of neurons of input layer */
    private double[] x;  // n-sized
    private final int n;

    /*  Activation of neurons of hidden layer */
    private double[] z;

    /*  Activation of neurons of output layer */
    private double[] y;

    /* Weights of connections of input layer to hidden layer */
    private double[][] v;

    /* Weights of connections of hidden layer to output layer */
    private double[][] w;

    /* Change of weights of connections that go from input layer to hidden layer */
    private double[][] dv;

    /* Change of weights of connections that go from hidden layer to output layer */
    private double[][] dw;

    /* Errors for output layer */
    private double[] dk;

    /* Errors for hidden layer */
    private double[] dj; 
    private final int p; // p-sized

    /* Output Target vector */
    private double[] t; 
    private final int m; // m-sized


    /* Learning rate */
    private double a;

    /* Treshold */
    private double treshold;

    /* */
    double[] z_in;
    double[] y_in;

    /* Max value for input values of weights and biases */
    private static final double MAX_INIT_VAL = 0.5;
    private static final double TOLERANCE = Math.pow(10, -3);
    private static int MAX_PERIOD_NUM;

    private final HashMap<Double, Double> fTable;

    private String OUT_PATH = ".";

    transient BufferedWriter bw;

    public MultilayerPerceptron(int inputNeurons, int hiddenNeurons, int outputNeurons, double learingRate, double treshold) {
        this(inputNeurons, hiddenNeurons, outputNeurons, learingRate, treshold, "./ecm.csv", 1000);
    }

    public MultilayerPerceptron(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate, double treshold, String path, int periods) {
        OUT_PATH = path;
        MAX_PERIOD_NUM = periods;

        n = inputNeurons;
        p = hiddenNeurons;
        m = outputNeurons;

        /* Create arrays for the activations */
        x = new double[n];
        z = new double[p];
        y = new double[m];

        z_in = new double[hiddenNeurons];
        y_in = new double[outputNeurons];

        /* Create arrays of weights of connections.
         * In both cases, the first index is saved for the bias.
         * For example, v[0][j] is the bias of the hidden neuron j. 
         */
        v = new double[inputNeurons + 1][hiddenNeurons]; // Incluimos el sesgo
        w = new double[hiddenNeurons + 1][outputNeurons]; // Incluimos el sesgo

        /* Create arrays of weights of connections.
         * In both cases, the first index is saved for the bias.
         * For example, v[0][j] is the bias of the output neuron k. 
         */
        dv = new double[inputNeurons + 1][hiddenNeurons]; // Include the bias
        dw = new double[hiddenNeurons + 1][outputNeurons]; // Include the bias

        /* Arrays to save the errors from the input layer (k) and the output layer (j)
         */
        dk = new double[outputNeurons];
        dj = new double[hiddenNeurons];

        /* Create the array for saving the expected outputs of the network at each iteration. 
         */
        t = new double[outputNeurons];

        /* Learning Rate */
        a = learningRate;

        /* Treshold */
        this.treshold = treshold;

        /* Initialize weights and biases */
        initializePerceptron(); // Step 0
        try {
            bw = new BufferedWriter(new FileWriter(OUT_PATH));
        } catch (IOException ex) {
            System.out.println("Error with file path");
        }

        fTable = new HashMap<>();
//        initFTable();
    }

    public MultilayerPerceptron(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate) {
        this(inputNeurons, hiddenNeurons, outputNeurons, learningRate, -1);
    }

    @Override
    public void training(Data trainDate) {
        boolean shouldStop = false;
        double mse;
        double last = 1;
        int counter = 0;
        try {
            // MSE = Mean Squared Error
            bw.write("Period,MSE\n");
        } catch (IOException ex) {
            Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
        }

        while (!shouldStop && counter < MAX_PERIOD_NUM) { // Step 1
            counter++;
            /* Iterate through the training sets */
            double accumulatedError = 0;
            double numErrors = 0;
            for (Double[] s : trainDate.getData()) {  // Step 2

                feedForward(s); // Steps 3, 4 y 5.

                backPropagation();

                /* Update weights and biases*/
                updateWeights();

                accumulatedError += outputSquaredError();

                if (s[trainDate.getClassIndex()] != getTargetClass()) {
                    numErrors++;
                }

            }


            mse = accumulatedError / m / trainDate.getDataCount();
            
            double ratioError = Math.abs(1d - mse / last);
            
            
            System.out.println("PERIOD: " + counter + ", Errors:" + numErrors + ", Ratio:"+ratioError);
//            if (numErrors == 0) {
//                shouldStop = true;
//            }


            if (ratioError < TOLERANCE) {
                shouldStop = true;
            }
            last = mse;
            double porcErr = (double) numErrors / trainDate.getDataCount();
            try {
                bw.write(counter + "," + mse + "\n");
            } catch (IOException ex) {
                Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        System.out.println("PERIOD COUNT: " + counter);

        try {
            bw.close();
        } catch (IOException ex) {
            Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public ArrayList<Integer> classify(Data testData) {
        ArrayList<Integer> clases = new ArrayList<>();
        for (Double[] s : testData.getData()) {
            int clasePred = classifyPattern(convertArray(s));
            clases.add(clasePred);
        }
        return clases;
    }

    private double randValue() {
        return (Math.random() * 2 * MAX_INIT_VAL) - MAX_INIT_VAL;
    }

    private int getTargetClass() {
        double max = -1 * Double.MAX_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < y.length; i++) {
            if (y[i] > max) {
                max = y[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
  

    /* Private methods */
    private void initializePerceptron() {
        /* Bias and weights of input layer to hidden layer */ 
        for (double[] v1 : v) {
            for (int i = 0; i < v1.length; i++) {
                v1[i] = randValue();
            }
        }
        /* Bias and weights from hidden layer to output layer */
        for (double[] w1 : w) {
            for (int i = 0; i < w1.length; i++) {
                w1[i] = randValue();
            }
        }
        
        /* Changes on biases and weights from the input layer to the hidden layer */
        for (double[] dv1 : dv) {
            for (int i = 0; i < dv1.length; i++) {
                dv1[i] = 0;
            }
        }
        
        /* Changes on the bias and weights from the hidden layer to the output layer */
        for (double[] dw1 : dw) {
            for (int i = 0; i < dw1.length; i++) {
                dw1[i] = 0;
            }
        }
        /* Errors on the output layer */
        for (int i = 0; i < dk.length; i++) {
            dk[i] = 0;
        }
        /* Errors on the hidden layer */
        for (int i = 0; i < dj.length; i++) {
            dj[i] = 0;
        }
    }

    private void calculateHiddenLayerActivations() {
        for (int j = 0; j < p; j++) {
            z_in[j] = v[0][j];
            for (int i = 0; i < n; i++) {
                z_in[j] += x[i] * v[i + 1][j];
            }
            z[j] = f(z_in[j]);
        }
    }

    private void calculateOuputLayerActivations() {
        for (int k = 0; k < m; k++) {
            y_in[k] = w[0][k];
            for (int j = 0; j < p; j++) {
                y_in[k] += z[j] * w[j + 1][k];
            }
            y[k] = f(y_in[k]);
        }
    }

    /**
     * Sigmoidal function
     *
     * @param x
     * @return sigmoidal function of x
     */
    private double f(double x) {
        double result = (double) (2 / (1 + Math.exp(-x))) - 1;
        return result;

    }

    private static double fRaw(double x) {
        double result = (double) (2 / (1 + Math.exp(-x))) - 1;
        return result;
    }

    private double fp(double x) {
        return 0.5 * (1 + f(x)) * (1 - f(x));
    }

    private void calculateOutputLayerError() {
        for (int k = 0; k < m; k++) {
            dk[k] = (t[k] - y[k]) * fp(y_in[k]);
            /* Bias */
            dw[0][k] = a * dk[k];
            /* Change on weights */
            for (int j = 1; j <= z.length; j++) {
                dw[j][k] = a * dk[k] * z[j - 1];  // revisar
            }
        }
    }

    private void calculateHiddenLayerError() {
        double[] dj_in = new double[p];
        for (int j = 0; j < p; j++) {
            dj_in[j] = 0;
            for (int k = 0; k < m; k++) {
                dj_in[j] += dk[k] * w[j + 1][k];
            }
            dj[j] = dj_in[j] * fp(z_in[j]);

            /* Calculate change on bias */
            dv[0][j] = a * dj[j];

            /* Calculate change on connections */
            for (int i = 1; i <= x.length; i++) {
                dv[i][j] = a * dj[j] * x[i - 1]; // revisar
            }
        }
    }

    private void setExpectedOutput(double d) {
        int idx = (int) d;
        for (int i = 0; i < t.length; i++) {
            t[i] = -.9;
        }
        t[idx] = .9;
    }

    private void updateWeights() {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j <= p; j++) {
                w[j][k] += dw[j][k];
            }
        }
        for (int j = 0; j < p; j++) {
            for (int i = 0; i <= n; i++) {
                v[i][j] += dv[i][j];
            }
        }
    }

    private double outputSquaredError() {
        double accumulatedError = 0;
        for (int i = 0; i < m; i++) {
            accumulatedError += Math.pow((t[i] - y[i]), 2);
        }
//        accumulatedError = accumulatedError / m;
        return accumulatedError;
    }
    private void partialFeedForward(Double[] s) {
        /* Feedforward */
        for (int i = 0; i < n; i++) { // Paso 3
            x[i] = s[i];
        }
        /* Feedforward
           Calculate response of neurons of the hidden layer (z)
         */
        calculateHiddenLayerActivations(); // Step 4
        /* Calculate response of neurons of the output layer (y) */
        calculateOuputLayerActivations(); // Step 5     
    }
    private void feedForward(Double[] s) {
        partialFeedForward(s);
        /* Asign objective vector */
        setExpectedOutput(s[s.length - 1]);
    }

    private void backPropagation() {
        /* Backpropagation
           Calculate the error on the output layer (y)
         */
        calculateOutputLayerError(); // Step 6
        /* Calculate error on the hidden layer */
        calculateHiddenLayerError(); // Step 7
    }

    private String getAllWeights() {
        String r = "----------------\nALL WEIGHTS\n";
        for (double[] v1 : v) {
            for (int i = 0; i < v1.length; i++) {
                r += v1[i] + ",";
            }
        }
        r += "\n";
        /* Bias and weights of the hidden layer to the output layer */
        for (double[] w1 : w) {
            for (int i = 0; i < w1.length; i++) {
                r += w1[i] + ",";
            }
        }
        r += "\n-----------------------";
        return r;
    }
    
    @Override
    public int classifyPattern(double[] s) {
        partialFeedForward(convertArray(s));
        return getTargetClass();
    }

   @Override
    public double getScore(double[] s) {
        partialFeedForward(convertArray(s));
        return y[1];
    }

    private void initFTable() {
        System.out.println("Initializing fTable");
        for (double i = -30; i <= 30; i += 0.00001) {
            i = setPrecision(i, 5);
//            System.out.println(i);
            fTable.put(i, fRaw(i));
        }
        System.out.println("fTable initialized");
    }
    
    private double setPrecision(double d, int p){
        return new BigDecimal(d)
                .setScale(p, BigDecimal.ROUND_HALF_UP)
                .doubleValue();
    }

    private Double[] convertArray(double[] s) {
        Double[] res = new Double[s.length];
        for (int i = 0; i < s.length; i++) {
            res[i] = s[i];
        }
        return res;
    }

    private double[] convertArray(Double[] s) {
        double[] res = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            res[i] = s[i];
        }
        return res;
    }

    

}
