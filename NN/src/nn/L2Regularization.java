package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class L2Regularization extends AbstractRegularization{
    
    private double lambda;
    
    public L2Regularization(double lambda){
        this.lambda = lambda;
    }
    
    public double regularizeCost(Matrix weight, boolean bias){
        if(bias)
            return MTJMathExt.sum(MTJCreateExt.toVector(MTJOpExt.powExtend(MTJCreateExt.splitMatrix(weight, 0, -1, 1, -1), MTJCreateExt.single(2))), 2).get(0,0)*0.5*this.lambda;
        else
            return MTJMathExt.sum(MTJCreateExt.toVector(MTJOpExt.powExtend(weight, MTJCreateExt.single(2))), 2).get(0,0)*0.5*this.lambda;
    }
    
    public Matrix regularizeGradient(Matrix weight, boolean bias){
        if(bias)
            return MTJConcat.concat(MTJCreateExt.Const(weight.numRows(),1,0), MTJOpExt.timesExtend(MTJCreateExt.splitMatrix(weight, 0, -1, 1, -1), MTJCreateExt.single(lambda)), 2).scale(lambda);
        else
            return MTJOpExt.timesExtend(weight, MTJCreateExt.single(lambda)).scale(lambda);
    }
    
}
