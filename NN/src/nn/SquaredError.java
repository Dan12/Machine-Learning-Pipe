package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class SquaredError extends AbstractCostFunction{
    
    public SquaredError(){}
    
    // get the cost of activations a given the target t
    public double getCost(Matrix a, Matrix t){
        return 0.0;
    }
    
    // get error of a given target t
    public Matrix getError(Matrix a, Matrix t){
        return null;
    }
}