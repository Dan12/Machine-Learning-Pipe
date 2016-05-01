package nn;

import no.uib.cipr.matrix.*;

public class LogLikelihoodCost extends AbstractCostFunction{
    
    public LogLikelihoodCost(){}
    
    // get the cost of activations a given the target t
    public double getCost(Matrix a, Matrix t){
        return -1*MTJMathExt.sum(MTJMathExt.sum(MTJOpExt.logExtend(MTJOpExt.timesExtend(a,t)), 2), 1).get(0,0);
    }
    
    // get error of a given target t
    public Matrix getError(Matrix a, Matrix t, AbstractActivationFunction activation){
        return MTJOpExt.minusExtend(a, t);
    }
    
}