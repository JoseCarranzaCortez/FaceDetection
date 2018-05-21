/**
 * @author José Carranza
 */
package splitting;

import java.util.ArrayList;

public class Partition {

    private ArrayList<Integer> indicesTrain;
    private ArrayList<Integer> indicesTest;

    public Partition(ArrayList<Integer> indTrain, ArrayList<Integer> indTest) {
        indicesTrain = indTrain;
        indicesTest = indTest;
    }

    public ArrayList<Integer> getIndicesTrain() {
        return indicesTrain;
    }

    public ArrayList<Integer> getIndicesTest() {
        return indicesTest;
    }

    @Override
    public String toString() {
        return "[Test: ["+indicesTest+"], Train: ["+indicesTrain+"]]";
    }
}
