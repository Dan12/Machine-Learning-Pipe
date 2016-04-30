package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public abstract class AbstractCostFunction{
    
    // get the cost of activations a given the target t
    abstract double getCost(Matrix a, Matrix t);
    
    // get error of a given target t
    abstract Matrix getError(Matrix a, Matrix t);
}