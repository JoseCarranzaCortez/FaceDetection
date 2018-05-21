/**
 * @author José Carranza
 */
package splitting;

import data.Data;
import java.util.ArrayList;

public interface SplittingStrategy {
	public String getNombreEstrategia();
	public int getNumeroParticiones();
	public ArrayList<Partition> crearParticiones(Data datos);
}