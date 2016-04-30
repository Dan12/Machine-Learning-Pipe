package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

abstract class AbstractRegularization{
    
    abstract double regularizeCost(Matrix weight, boolean bias);
    
    abstract Matrix regularizeGradient(Matrix weight, boolean bias);
}