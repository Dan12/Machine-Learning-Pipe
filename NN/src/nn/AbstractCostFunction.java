package nn;

import no.uib.cipr.matrix.*;

public abstract class AbstractCostFunction{
    
    // get the cost of activations a given the target t
    abstract double getCost(Matrix a, Matrix t);
    
    // get error of a given target t
    abstract Matrix getError(Matrix a, Matrix t, AbstractActivationFunction activation);
}