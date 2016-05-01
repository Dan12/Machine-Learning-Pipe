package nn;

import no.uib.cipr.matrix.*;

public class SquaredError extends AbstractCostFunction{
    
    public SquaredError(){}
    
    // get the cost of activations a given the target t
    public double getCost(Matrix a, Matrix t){
        return 0.5*(MTJMathExt.sum(MTJMathExt.sum(MTJOpExt.powExtend(MTJOpExt.minusExtend(a,t), MTJCreateExt.single(2)),1),2)).get(0,0);
    }
    
    // get error of a given target t
    public Matrix getError(Matrix a, Matrix t, AbstractActivationFunction activation){
        return MTJOpExt.timesExtend(MTJOpExt.minusExtend(a, t), activation.getActivationDerivative(a));
    }
}