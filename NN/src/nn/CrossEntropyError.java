package nn;

import no.uib.cipr.matrix.*;

public class CrossEntropyError extends AbstractCostFunction{
    
    public CrossEntropyError(){}
    
    // get the cost of activations a given the target t
    public double getCost(Matrix a, Matrix t){
        return MTJMathExt.sum(MTJOpExt.minusExtend(
            MTJMathExt.sum(MTJOpExt.timesExtend(new DenseMatrix(t,true).scale(-1), MTJOpExt.logExtend(a)), 1), 
            MTJMathExt.sum(MTJOpExt.timesExtend(MTJOpExt.minusExtend(MTJCreateExt.single(1), t), MTJOpExt.logExtend(MTJOpExt.minusExtend(MTJCreateExt.single(1), a))), 1)), 
        2).get(0,0);
    }
    
    // get error of a given target t
    public Matrix getError(Matrix a, Matrix t, AbstractActivationFunction activation){
        return MTJOpExt.minusExtend(a, t);
    }
}