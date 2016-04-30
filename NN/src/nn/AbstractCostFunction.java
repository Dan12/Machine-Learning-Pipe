package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public abstract class AbstractCostFunction{
    
    // get the cost of activations a given the target t
    abstract double getCost(Matrix a, Matrix t);
}