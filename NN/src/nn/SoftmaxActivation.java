package nn;

import no.uib.cipr.matrix.*;

public class SoftmaxActivation extends AbstractActivationFunction{
    
    public SoftmaxActivation(){}
    
    // returns f(z) where f is the activation function
    public Matrix getActivation(Matrix z){
        return MTJOpExt.divideExtend(MTJOpExt.powExtend(MTJCreateExt.single(Math.E), z), MTJMathExt.sum(MTJMathExt.sum(MTJOpExt.powExtend(MTJCreateExt.single(Math.E), z), 2), 1));
    }
    
    // returns f'(z) where f is the activation function
    public Matrix getRawDerivative(Matrix z){
        return MTJCreateExt.Ones(z.numRows(), z.numColumns());
    }
    
    // returns g'(a) where a is activations and g is simplified derivative
    public Matrix getActivationDerivative(Matrix a){
        return MTJCreateExt.Ones(a.numRows(), a.numColumns());
    }
}