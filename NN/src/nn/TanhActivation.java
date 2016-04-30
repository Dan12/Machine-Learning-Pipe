package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class TanhActivation extends AbstractActivationFunction{
    
    public TanhActivation(){}
    
    // returns f(z) where f is the tanh function
    public Matrix getActivation(Matrix z){
        Matrix tempZ2 = new DenseMatrix(z,true);
        tempZ2.scale(2);
        return MTJOpExt.divideExtend(MTJOpExt.minusExtend(MTJOpExt.powExtend(MTJCreateExt.single(Math.E), tempZ2), MTJCreateExt.single(1)), MTJOpExt.plusExtend(MTJOpExt.powExtend(MTJCreateExt.single(Math.E), tempZ2), MTJCreateExt.single(1)));
    }
    
    // returns f'(z) where f is the tanh function
    public Matrix getDerivative(Matrix z){
        return MTJOpExt.minusExtend(MTJCreateExt.single(1), MTJOpExt.powExtend(getActivation(z), MTJCreateExt.single(2)));
    }
    
    // returns g'(a)
    public Matrix getSpecialDerivative(Matrix a){
        return MTJOpExt.minusExtend(MTJCreateExt.single(1), MTJOpExt.powExtend(a, MTJCreateExt.single(2)));
    }
}