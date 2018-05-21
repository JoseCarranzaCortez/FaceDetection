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

    private int numAtributos = 0;
    private int numClases = 0;

    ArrayList<Double[]> datos = new ArrayList<>();

    public Data(int numAtributos, int numClases) {
        this.numClases = numClases;
        this.numAtributos = numAtributos;
    }

    public void addDato(Double[] dato) {
        datos.add(dato);
    }

    public int getNumDatos() {
        return this.datos.size();
    }

    public int getNumAtributos() {
        return this.numAtributos;
    }

    public int getNumClases() {
        return numClases;
    }

    public ArrayList<Double[]> getDatos() {
        return datos;
    }

    public int getClassIndex() {
        return this.numAtributos;
    }

    public Map<Integer, Double> getProbablidadesAPriori() {
        double aux;
        double count;
        int nAtributos = this.numAtributos;
        Map<Integer, Double> m = new HashMap<>();

        //Calculamos la frecuencia de las clases
        for (int i = 0; i < this.getNumDatos(); i++) {
            aux = datos.get(i)[nAtributos];
            count = m.containsKey(aux) ? m.get(aux) : 0;
            m.put((int) aux, count + 1);
        }
        //Calculamos las probabilidades (freq/NumDatos)
        for (Map.Entry<Integer, Double> entrySet : m.entrySet()) {
            Double value = entrySet.getValue();
            entrySet.setValue(value / getNumDatos());
        }
        return m;
    }

//// Devuelve un objeto tipo Data con los datos de clase 'c'
//
//    public Data getDatosByClase(int c) {
//        ArrayList<Integer> l = new ArrayList<>();
//        int current;
//        Data d;
//        //Selecciono los ids de los datos de clase c
//        for (int i = 0; i < this.getNumDatos(); i++) {
//            current = (int) datos[i][this.getNumAtributos() - 1];
//            if (current == c) {
//                l.add(i);
//            }
//        }
//        d = new Data(l.size(), tipoAtributos);
//        for (int i = 0; i < d.datos.length; i++) {
//            d.datos[i] = this.datos[l.get(i)];
//        }
//        return d;
//    }
//
    public Data extraeDatosTrain(Partition idx) {
        int nDatos = idx.getIndicesTrain().size();
        Data d = new Data(this.numAtributos, this.numClases);
        d.datos = new ArrayList<>();

        for (int i = 0; i < nDatos; i++) {
            d.datos.add(this.datos.get(idx.getIndicesTrain().get(i)));
        }

        return d;
    }

    public Data extraeDatosTest(Partition idx) {
        int nDatos = idx.getIndicesTest().size();
        Data d = new Data(this.numAtributos, this.numClases);
        d.datos = new ArrayList<>();

        for (int i = 0; i < nDatos; i++) {
            d.datos.add(this.datos.get(idx.getIndicesTest().get(i)));
        }

        return d;
    }

    @Override
    public String toString() {
        String r = "";
        for (Double[] dato : datos) {
            r += "[";
            for (Double dato1 : dato) {
                r += dato1 + " ";
            }
            r += "]\n";
        }
        return r;
    }

    public static Data cargaDeFichero(String nombreDeFichero) throws IOException {
        return cargaDeFichero(nombreDeFichero, false);
    }

    public static Data cargaDeFichero(String nombreDeFichero, boolean normalizar) throws IOException {
        int numAtributos = 0;
        /* Total de elementos en el fichero */

        int numClases = 0;
        Data respuesta;
        try {
            FileReader fr = new FileReader(nombreDeFichero);
            BufferedReader br = new BufferedReader(fr);

            String line = br.readLine();
            try {
                String[] vals = line.split(" ");
                numAtributos = Integer.parseInt(vals[0]);
                numClases = Integer.parseInt(vals[1]);
            } catch (NumberFormatException nfe) {
                System.out.println("Número de datos y de clases debe de ser numérico.");
            }

            line = br.readLine();
            int x = 0;
            int y = 0;

            respuesta = new Data(numAtributos, numClases);

            while (line != null) {

                /* Cargar los datos */
                String[] atributos = line.split("\\s+");

                /* Ignorar si no es el número exacto de atributos */
                if (atributos.length == numAtributos + numClases) {
                    Double[] d = new Double[numAtributos + 1];
                    double clase = 0;
                    for (String str : line.split("\\s+")) {
                        try {
                            if (x < numAtributos) {
                                d[x] = Double.parseDouble(str);
                            } else if (Double.parseDouble(str) == -100) {
                                d[numAtributos] = clase;
                            } else if (Double.parseDouble(str) == 1) {
                                d[numAtributos] = clase;
                            } else {
                                clase++;
                                continue;
                            }
                        } catch (NumberFormatException nfe) {
                            System.out.println("Error en línea " + y + ", atributo " + x
                                    + ". String: \" " + str + "\" no es continuo." + nfe.getMessage());
                        }
                        x++;
                    }
                    x = 0;
                    y++;

                    respuesta.datos.add(d);
                }
                line = br.readLine();
            }
            if (normalizar) {
                return respuesta.normaliza();
            }
            return respuesta;
        } catch (FileNotFoundException fnfe) {
            System.out.println("Error leyendo archivo de entrada");
        }
        return null;
    }

    private void normalizar2() {
        for (int i = 0; i < this.getNumAtributos(); i++) {
            double max = 0;
            double min = Double.MAX_VALUE;
            for (int j = 0; j < this.datos.size(); j++) {
                double d = this.datos.get(j)[i];
                if (d > max) {
                    max = d;
                }
                if (d < min) {
                    min = d;
                }
            }
            for (int j = 0; j < this.datos.size(); j++) {
                this.datos.get(j)[i] = (this.datos.get(j)[i] - min) * 1 / (max - min);
            }
        }
    }

    /**
     * Metodo que calcula la media para todos los atributos dado un grupo de datos. Si se requiere
     * sacar por clase, llamar primero al metodo getDatosByClase y aplicar getAttMeans() sobre el
     * nuevo objeto.
     */
    /**
     * Metodo que calcula la media para todos los atributos dado un grupo de datos.Si se requiere
     * sacar por clase, llamar primero al metodo getDatosByClase y aplicar getAttMeans() sobre el
     * nuevo objeto.
     *
     * @return
     */
    public ArrayList<Double> getAttMeans() {
        double valores[] = new double[getNumAtributos()];
        ArrayList<Double> medias = new ArrayList<>();

        for (Double[] atr : datos) {
            for (int j = 0; j < getNumAtributos(); j++) {
                valores[j] += atr[j];
            }
        }
        for (int i = 0; i < valores.length; i++) {
            valores[i] /= getNumDatos();
            medias.add(valores[i]);
        }

        return medias;
    }

    /**
     * Metodo que calcula la varianza para todos los atributos dado un grupo de datos. Si se
     * requiere sacar por clase, llamar primero al metodo getDatosByClase y aplicar getAttMeans()
     * sobre el nuevo objeto.
     *
     * @return
     */
    public ArrayList<Double> getAttStandardDeviation() {
        double valores[] = new double[getNumAtributos()];
        ArrayList<Double> medias = this.getAttMeans();
        ArrayList<Double> desviaciones = new ArrayList<>();

        for (Double[] atr : datos) {
            for (int j = 0; j < getNumAtributos(); j++) {
                valores[j] += (atr[j] - medias.get(j)) * (atr[j] - medias.get(j));
            }
        }

        for (int i = 0; i < valores.length; i++) {
            valores[i] = valores[i] / getNumDatos();
            desviaciones.add(valores[i]);
        }

        return desviaciones;
    }

    public Data normaliza() {
        ArrayList<Double> means = this.getAttMeans();
        ArrayList<Double> sd = this.getAttStandardDeviation();

        Data res = new Data(this.getNumAtributos(), this.getNumClases());

        Double aux[];
        for (int i = 0; i < this.getNumDatos(); i++) {
            aux = new Double[this.getNumAtributos() + 1];
            for (int j = 0; j < (this.getNumAtributos() + 1); j++) {
                if (j == this.getNumAtributos()) {
                    aux[j] = this.datos.get(i)[j];
                } else if (sd.get(j) != 0) {
//                        aux[j] = (this.getDatos().get(i)[j] - means.get(j)) / sd.get(j);
                    aux[j] = (this.getDatos().get(i)[j]) / 255;
                } else {
                    aux[j] = (this.getDatos().get(i)[j]) / 255;
//                        aux[j] = (this.getDatos().get(i)[j] - means.get(j)) / Double.MIN_VALUE;
                } //                    System.out.println(sd.get(j));
            }

//            System.out.println("");
            res.datos.add(aux);
        }
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter("./data/estadistica.txt"));
            bw.write("Medias: " + means + "\n");
            bw.write("Desviaciones:" + sd + "\n");
            bw.close();

            bw = new BufferedWriter(new FileWriter("./data/archivo_normalizado.txt"));
            bw.write(this.getNumAtributos() + " " + this.getNumClases() + "\n");
            for (Double[] dato : res.getDatos()) {
                for (Double d : dato) {
                    bw.write(d + " ");
                }
                bw.write("\n");
            }
        } catch (Exception e) {

        }

        return res;
    }
//
//    public Map<Integer, Set<Double>> getValueSet() {
//        Map<Integer, Set<Double>> values = new HashMap();
//        for (int i = 0; i < getNumAtributos(); i++) {
//            Set<Double> current = new HashSet<>();
//            for (int j = 0; j < getNumDatos(); j++) {
//                double val = this.datos[j][i];
//                current.add(val);
//            }
//            values.put(i, current);
//        }
//        return values;
//    }
}
