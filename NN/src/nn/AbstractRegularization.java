package nn;

import no.uib.cipr.matrix.*;

abstract class AbstractRegularization{
    
    abstract double regularizeCost(Matrix weight, boolean bias);
    
    abstract Matrix regularizeGradient(Matrix weight, boolean bias);
}