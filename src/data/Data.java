/**
 * @author José Carranza
 */
package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import splitting.Partition;
import java.util.HashMap;
import java.util.Map;

public class Data {

    private int attributeCount = 0;
    private int classCount = 0;

    ArrayList<Double[]> data = new ArrayList<>();

    public Data(int attributeCount, int classCount) {
        this.classCount = classCount;
        this.attributeCount = attributeCount;
    }

    public void addData(Double[] data) {
        this.data.add(data);
    }

    public int getDataCount() {
        return this.data.size();
    }

    public int getAttributeCount() {
        return this.attributeCount;
    }

    public int getClassCount() {
        return classCount;
    }

    public ArrayList<Double[]> getData() {
        return data;
    }

    public int getClassIndex() {
        return this.attributeCount;
    }

    public Map<Integer, Double> getAPrioriProbabilities() {
        double aux;
        double count;
        int nAtributos = this.attributeCount;
        Map<Integer, Double> m = new HashMap<>();

        // Claculate class frequency
        for (int i = 0; i < this.getDataCount(); i++) {
            aux = data.get(i)[nAtributos];
            count = m.containsKey(aux) ? m.get(aux) : 0;
            m.put((int) aux, count + 1);
        }
        // Calculate probabilities (freq/dataCount)
        for (Map.Entry<Integer, Double> entrySet : m.entrySet()) {
            Double value = entrySet.getValue();
            entrySet.setValue(value / getDataCount());
        }
        return m;
    }

    public Data extractTrainData(Partition idx) {
        int nDatos = idx.getIndicesTrain().size();
        Data d = new Data(this.attributeCount, this.classCount);
        d.data = new ArrayList<>();

        for (int i = 0; i < nDatos; i++) {
            d.data.add(this.data.get(idx.getIndicesTrain().get(i)));
        }

        return d;
    }

    public Data extractTestData(Partition idx) {
        int nDatos = idx.getIndicesTest().size();
        Data d = new Data(this.attributeCount, this.classCount);
        d.data = new ArrayList<>();

        for (int i = 0; i < nDatos; i++) {
            d.data.add(this.data.get(idx.getIndicesTest().get(i)));
        }

        return d;
    }

    @Override
    public String toString() {
        String r = "";
        for (Double[] dato : data) {
            r += "[";
            for (Double dato1 : dato) {
                r += dato1 + " ";
            }
            r += "]\n";
        }
        return r;
    }

    public static Data loadFile(String filePath) throws IOException {
        return loadFile(filePath, false);
    }

    public static Data loadFile(String filePath, boolean normalize) throws IOException {
        int totalAttribuets = 0;
        /* Total elements on the file */

        int totalClasses = 0;
        Data resposne;
        try {
            FileReader fr = new FileReader(filePath);
            BufferedReader br = new BufferedReader(fr);

            String line = br.readLine();
            try {
                String[] vals = line.split(" ");
                totalAttribuets = Integer.parseInt(vals[0]);
                totalClasses = Integer.parseInt(vals[1]);
            } catch (NumberFormatException nfe) {
                System.out.println("Class count and attribute count must be numeric.");
            }

            line = br.readLine();
            int x = 0;
            int y = 0;

            resposne = new Data(totalAttribuets, totalClasses);

            while (line != null) {

                /* Load data */
                String[] atributos = line.split("\\s+");

                /* Ignore if the line does not contain the exact number of attributes */
                if (atributos.length == totalAttribuets + totalClasses) {
                    Double[] d = new Double[totalAttribuets + 1];
                    double clase = 0;
                    for (String str : line.split("\\s+")) {
                        try {
                            if (x < totalAttribuets) {
                                d[x] = Double.parseDouble(str);
                            } else if (Double.parseDouble(str) == -100) {
                                d[totalAttribuets] = clase;
                            } else if (Double.parseDouble(str) == 1) {
                                d[totalAttribuets] = clase;
                            } else {
                                clase++;
                                continue;
                            }
                        } catch (NumberFormatException nfe) {
                            System.out.println("Error on line " + y + ", attribute " + x
                                    + ". String: \" " + str + "\"  is not continuous." + nfe.getMessage());
                        }
                        x++;
                    }
                    x = 0;
                    y++;

                    resposne.data.add(d);
                }
                line = br.readLine();
            }
            if (normalize) {
                return resposne.normaliza();
            }
            return resposne;
        } catch (FileNotFoundException fnfe) {
            System.out.println("Error reading input file.");
        }
        return null;
    }


    /**
     * Method that calculates the mean for all attributes given a data set. If its required to get 
     * per class, call first getDatosByClase and apply getAttMeans() to the new object.
     *
     * @return an ArrayList with attribute means
     */
    public ArrayList<Double> getAttMeans() {
        double valores[] = new double[getAttributeCount()];
        ArrayList<Double> medias = new ArrayList<>();

        for (Double[] atr : data) {
            for (int j = 0; j < getAttributeCount(); j++) {
                valores[j] += atr[j];
            }
        }
        for (int i = 0; i < valores.length; i++) {
            valores[i] /= getDataCount();
            medias.add(valores[i]);
        }

        return medias;
    }

    /**
     * Method that calculates the variance for all attributes given a data set. If its required to get 
     * per class, call first getDatosByClase and apply getAttMeans() to the new object.
     *
     * @return
     */
    public ArrayList<Double> getAttStandardDeviation() {
        double valores[] = new double[getAttributeCount()];
        ArrayList<Double> medias = this.getAttMeans();
        ArrayList<Double> desviaciones = new ArrayList<>();

        for (Double[] atr : data) {
            for (int j = 0; j < getAttributeCount(); j++) {
                valores[j] += (atr[j] - medias.get(j)) * (atr[j] - medias.get(j));
            }
        }

        for (int i = 0; i < valores.length; i++) {
            valores[i] = valores[i] / getDataCount();
            desviaciones.add(valores[i]);
        }

        return desviaciones;
    }

    public Data normaliza() {
        ArrayList<Double> means = this.getAttMeans();
        ArrayList<Double> sd = this.getAttStandardDeviation();

        Data res = new Data(this.getAttributeCount(), this.getClassCount());

        Double aux[];
        for (int i = 0; i < this.getDataCount(); i++) {
            aux = new Double[this.getAttributeCount() + 1];
            for (int j = 0; j < (this.getAttributeCount() + 1); j++) {
                if (j == this.getAttributeCount()) {
                    aux[j] = this.data.get(i)[j];
                } else if (sd.get(j) != 0) {
                    aux[j] = (this.getData().get(i)[j]) / 255;
                } else {
                    aux[j] = (this.getData().get(i)[j]) / 255;
                } 
            }
            res.data.add(aux);
        }
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter("./data/estadistica.txt"));
            bw.write("Medias: " + means + "\n");
            bw.write("Desviaciones:" + sd + "\n");
            bw.close();

            bw = new BufferedWriter(new FileWriter("./data/archivo_normalizado.txt"));
            bw.write(this.getAttributeCount() + " " + this.getClassCount() + "\n");
            for (Double[] dato : res.getData()) {
                for (Double d : dato) {
                    bw.write(d + " ");
                }
                bw.write("\n");
            }
        } catch (Exception e) {

        }

        return res;
    }
    
}
