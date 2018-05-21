/**
 * @author José Carranza
 */
package splitting;

import data.Data;
import java.util.ArrayList;

public class SimpleValidation implements SplittingStrategy {

    private final int numParticiones = 0;
    private final boolean USE_ALL_SAMPLES = true;
    private final double PORCENTAJE_TRAIN = .3;
    private final boolean ALEATORIO = true;

    @Override
    //Devuelve el nombre de la estrategia de particionaado
    public String getNombreEstrategia() {
        return "Validación Simple";
    }

    @Override
    //Devuelve el numero de particiones
    public int getNumeroParticiones() {
        return numParticiones;
    }

    @Override
    // Crea particiones segun el metodo tradicional de división según el
    // porcentaje deseado. Devuelve una sola partición con un conunto de train
    // y otro de test.
    public ArrayList<Partition> crearParticiones(Data datos) {
        ArrayList<Partition> particiones = new ArrayList<>();

        int total = datos.getNumDatos();
//        System.out.println("Total: "+total);

        int idx_train = (int) ((double) total * PORCENTAJE_TRAIN);

//        System.out.println("Indices train: "+idx_train);
        ArrayList<Integer> indices_train = new ArrayList<>();
        ArrayList<Integer> indices_test = new ArrayList<>();

        int[] indices = new int[total];

        for (int i = 0; i < total; i++) {
            indices[i] = i;
        }
        if (ALEATORIO) {
            for (int i = 0; i < total; i++) {
                int otherIdx = (int) (Math.random() * total);
                int aux = indices[i];
                indices[i] = indices[otherIdx];
                indices[otherIdx] = aux;
            }
        }

        if (USE_ALL_SAMPLES) {
            idx_train = total;
        }
        for (int i = 0; i < total; i++) {
            if (i < idx_train) {
                indices_train.add(indices[i]);
                if (USE_ALL_SAMPLES) {
                    indices_test.add(indices[i]);
                }
            } else {
                indices_test.add(indices[i]);
            }
        }

//        System.out.println("Train: "+indices_train);
//        System.out.println("Test: "+indices_test);
        Partition p = new Partition(indices_train, indices_test);

        particiones.add(p);

        return particiones;
    }
}
