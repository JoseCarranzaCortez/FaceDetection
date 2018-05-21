/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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

    /*  Activaciones de las neuronas de la capa de entrada */
    private double[] x;  // Tamaño n
    private final int n;

    /*  Acticaciones de las neuronas de la capa oculta */
    private double[] z;

    /*  Acticaciones de las neuronas de la capa de salida */
    private double[] y;

    /* Pesos de las conexiones de la capa de entrada a la capa oculta */
    private double[][] v;

    /* Pesos de las conexiones de la capa oculta a la capa de salida */
    private double[][] w;

    /* Cambios en los pesos de las conexiones de la capa de entrada a la oculta */
    private double[][] dv;

    /* Cambios en los pesos de las conexiones de la capa oculta a la capa de salida */
    private double[][] dw;

    /* Errores para la capa de salida */
    private double[] dk;

    /* Errores para la capa oculta */
    private double[] dj; // Tamaño p
    private final int p;

    /* Vector objetivo de salida  */
    private double[] t; // Tamaño m
    private final int m;


    /* Tasa de aprendizaje */
    private double a;

    /* Umbral */
    private double umbral;

    /* */
    double[] z_in;
    double[] y_in;

    /* Valor máximo para valores iniciales de pesos y sesgos*/
    private static final double MAX_INIT_VAL = 0.5;
    private static final double TOLERANCIA = Math.pow(10, -3);
    private static int NUM_MAX_EPOCAS;

    private final HashMap<Double, Double> fTable;

    private String OUT_PATH = ".";

    transient BufferedWriter bw;

    public MultilayerPerceptron(int neuronasDeEntrada, int neuronasOcultas, int neuronasDeSalida, double tasaDeAprendizaje, double umbral) {
        this(neuronasDeEntrada, neuronasOcultas, neuronasDeSalida, tasaDeAprendizaje, umbral, "./ecm.csv", 1000);
    }

    public MultilayerPerceptron(int neuronasDeEntrada, int neuronasOcultas, int neuronasDeSalida, double tasaDeAprendizaje, double umbral, String path, int epocas) {
        OUT_PATH = path;
        NUM_MAX_EPOCAS = epocas;

        n = neuronasDeEntrada;
        p = neuronasOcultas;
        m = neuronasDeSalida;

        /* Creamos arreglos de activaciones */
        x = new double[n];
        z = new double[p];
        y = new double[m];

        z_in = new double[neuronasOcultas];
        y_in = new double[neuronasDeSalida];

        /* Creamos arreglos de pesos de conexiones 
         * En ambos casos, el primer índice corresponde al sesgo. 
         * Por ejemplo, v[0][j] es el sesgo de a la neurona oculta j
         */
        v = new double[neuronasDeEntrada + 1][neuronasOcultas]; // Incluimos el sesgo
        w = new double[neuronasOcultas + 1][neuronasDeSalida]; // Incluimos el sesgo

        /* Creamos arreglos de cambios en los pesos de las conexiones
         * En ambos casos, el primer índice corresponde al sesgo. 
         * Por ejemplo, dv[0][k] es el sesgo de a la neurona de salida k
         */
        dv = new double[neuronasDeEntrada + 1][neuronasOcultas]; // Incluimos el sesgo
        dw = new double[neuronasOcultas + 1][neuronasDeSalida]; // Incluimos el sesgo

        /* Creamos arreglos para guardar los errores de la capa de entrada (k) y la capa de 
         * salida (j)
         */
        dk = new double[neuronasDeSalida];
        dj = new double[neuronasOcultas];

        /* Creamos el arreglo para guardar las salidas esperadas de la red en cada iteración.
         */
        t = new double[neuronasDeSalida];

        /* Tasa de aprendizaje */
        a = tasaDeAprendizaje;

        /* Umbral */
        this.umbral = umbral;

        /* Inicializamos los pesos y sesgos */
        inicializarPerceptron(); // Paso 0
        try {
            bw = new BufferedWriter(new FileWriter(OUT_PATH));
        } catch (IOException ex) {
            System.out.println("Error with file path");
        }

        fTable = new HashMap<>();
//        initFTable();
    }

    public MultilayerPerceptron(int neuronasDeEntrada, int neuronasOcultas, int neuronasDeSalida, double tasaDeAprendizaje) {
        this(neuronasDeEntrada, neuronasOcultas, neuronasDeSalida, tasaDeAprendizaje, -1);
    }

    @Override
    public void training(Data datosTrain) {
        boolean parar = false;
        double ecm;
        double last = 1;
        int counter = 0;
        try {
            bw.write("Epoca,ECM\n");
        } catch (IOException ex) {
            Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
        }

        while (!parar && counter < NUM_MAX_EPOCAS) { // Paso 1
            counter++;
            /* Iteramos por los conjuntos de training */
            double errorAcumulado = 0;
            double errores = 0;
            for (Double[] s : datosTrain.getDatos()) {  // Paso 2

                feedForward(s); // Pasos 3, 4 y 5.

                backPropagation();

                /* Actualizar pesos y sesgos */
                actualizarPesos();

                errorAcumulado += errorCuadraticoSalida();

                if (s[datosTrain.getClassIndex()] != getClase()) {
                    errores++;
                }

            }


            ecm = errorAcumulado / m / datosTrain.getNumDatos();
            
            double ratioError = Math.abs(1d - ecm / last);
            
            
            System.out.println("EPOCA: " + counter + ", errores:" + errores + ", ratio:"+ratioError);
//            if (errores == 0) {
//                parar = true;
//            }


            if (ratioError < TOLERANCIA) {
                parar = true;
            }
            last = ecm;
            double porcErr = (double) errores / datosTrain.getNumDatos();
            try {
                bw.write(counter + "," + ecm + "\n");
            } catch (IOException ex) {
                Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        System.out.println("NUMERO DE EPOCAS: " + counter);

        try {
            bw.close();
        } catch (IOException ex) {
            Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public ArrayList<Integer> classify(Data datosTest) {
        ArrayList<Integer> clases = new ArrayList<>();
        for (Double[] s : datosTest.getDatos()) {
            int clasePred = classifyPattern(convertArray(s));
            clases.add(clasePred);
        }
        return clases;
    }

    private double randValue() {
        return (Math.random() * 2 * MAX_INIT_VAL) - MAX_INIT_VAL;
    }

    private int getClase() {
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
    private void inicializarPerceptron() {
        /* Sesgo y pesos de la capa de entrada a la capa oculta */
        for (double[] v1 : v) {
            for (int i = 0; i < v1.length; i++) {
                v1[i] = randValue();
            }
        }
        /* Sesgo y pesos de la capa oculta a la capa de salida */
        for (double[] w1 : w) {
            for (int i = 0; i < w1.length; i++) {
                w1[i] = randValue();
            }
        }
        /* Cambios en el sesgo y pesos de la capa de entrada a la capa oculta */
        for (double[] dv1 : dv) {
            for (int i = 0; i < dv1.length; i++) {
                dv1[i] = 0;
            }
        }
        /* Cambios en el sesgo y pesos de la capa oculta a la capa de salida */
        for (double[] dw1 : dw) {
            for (int i = 0; i < dw1.length; i++) {
                dw1[i] = 0;
            }
        }
        /* Errores en la capa de salida */
        for (int i = 0; i < dk.length; i++) {
            dk[i] = 0;
        }
        /* Errores en la capa oculta */
        for (int i = 0; i < dj.length; i++) {
            dj[i] = 0;
        }
    }

    private void calcularActivacionesCapaOculta() {
        for (int j = 0; j < p; j++) {
            z_in[j] = v[0][j];
            for (int i = 0; i < n; i++) {
                z_in[j] += x[i] * v[i + 1][j];
            }
            z[j] = f(z_in[j]);
        }
    }

    private void calcularActivacionesCapaDeSalida() {
        for (int k = 0; k < m; k++) {
            y_in[k] = w[0][k];
            for (int j = 0; j < p; j++) {
                y_in[k] += z[j] * w[j + 1][k];
            }
            y[k] = f(y_in[k]);
        }
    }

    /**
     * Función sigmoidal
     *
     * @param x
     * @return funcion sigmoideal de x
     */
    private double f(double x) {
        
//        x = setPrecision(x, 5);
        
//        if (fTable.containsKey(x)) {
//            return fTable.get(x);
//        } else {
            double result = (double) (2 / (1 + Math.exp(-x))) - 1;
//            System.out.println("--------------X:" + x);
//            fTable.put(x, result);
            return result;
//        }
    }

    private static double fRaw(double x) {
//        System.out.println("x: "+x);
        double result = (double) (2 / (1 + Math.exp(-x))) - 1;
        return result;
    }

    private double fp(double x) {
        return 0.5 * (1 + f(x)) * (1 - f(x));
    }

    private void calcularErrorCapaDeSalida() {
        for (int k = 0; k < m; k++) {
            dk[k] = (t[k] - y[k]) * fp(y_in[k]);
            /* Sesgo */
            dw[0][k] = a * dk[k];
            /* Cambios en los pesos*/
            for (int j = 1; j <= z.length; j++) {
                dw[j][k] = a * dk[k] * z[j - 1];  // revisar
            }
        }
    }

    private void calcularErrorCapaOculta() {
        double[] dj_in = new double[p];
        for (int j = 0; j < p; j++) {
            dj_in[j] = 0;
            for (int k = 0; k < m; k++) {
                dj_in[j] += dk[k] * w[j + 1][k];
            }
            dj[j] = dj_in[j] * fp(z_in[j]);

            /* Calculamos cambio en el sesgo */
            dv[0][j] = a * dj[j];

            /* Calculamos cambio en las conexiones */
            for (int i = 1; i <= x.length; i++) {
                dv[i][j] = a * dj[j] * x[i - 1]; // revisar
            }
        }
    }

    private void setSalidaEsperada(double d) {
        int idx = (int) d;
        for (int i = 0; i < t.length; i++) {
            t[i] = -.9;
        }
        t[idx] = .9;
    }

    private void actualizarPesos() {
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

//        for (int k = 0; k < y.length; k++) {
//            for (int j = 0; j < z.length; j++) {
//                w[j][k] += dw[j][k];
//                for (int i = 0; i < x.length; i++) {
//                    v[i][j] += dv[i][j];
//                }
//            }
//        }

        /*
         for (int k = 0; k < y.length; k++) {
            w[0][k] += dw[0][k];
            for (int j = 1; j <= z.length; j++) {
                w[j][k] += dw[j][k];
                v[0][j - 1] += dv[0][j - 1];
                for (int i = 1; i <= x.length; i++) {
                    v[i][j - 1] += dv[i][j - 1];
                }
            }
        }
         */
    }

    private double errorCuadraticoSalida() {
        double eAcumulado = 0;
        for (int i = 0; i < m; i++) {
            eAcumulado += Math.pow((t[i] - y[i]), 2);
        }
//        eAcumulado = eAcumulado / m;
        return eAcumulado;
    }
    private void feedForwardParcial(Double[] s) {
        /* Feedforward */
        for (int i = 0; i < n; i++) { // Paso 3
            x[i] = s[i];
        }
        /* Feedforward
           Calcular respuesta de las neuronas de la capa oculta (z) 
         */
        calcularActivacionesCapaOculta(); // Paso 4
        /* Calcular respuesta de las neuronas de la capa de salida (y)*/
        calcularActivacionesCapaDeSalida(); // Paso 5     
    }
    private void feedForward(Double[] s) {
        feedForwardParcial(s);
        /* Asignar vector objetivo */
        setSalidaEsperada(s[s.length - 1]);
    }

    private void backPropagation() {
        /* Backpropagation
           Calcular el error en la capa de salida (y)
         */
        calcularErrorCapaDeSalida(); // Paso 6
        /* Calcular el error en la capa oculta */
        calcularErrorCapaOculta(); // Paso 7
    }

    private String getAllWeights() {
        String r = "----------------\nALL WEIGHTS\n";
        for (double[] v1 : v) {
            for (int i = 0; i < v1.length; i++) {
                r += v1[i] + ",";
            }
        }
        r += "\n";
        /* Sesgo y pesos de la capa oculta a la capa de salida */
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
        feedForwardParcial(convertArray(s));
        return getClase();
    }

   @Override
    public double getScore(double[] s) {
        feedForwardParcial(convertArray(s));
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
