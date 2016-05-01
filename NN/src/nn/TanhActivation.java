package nn;

import no.uib.cipr.matrix.*;

public class TanhActivation extends AbstractActivationFunction{
    
    public TanhActivation(){}
    
    // returns f(z) where f is the tanh function
    public Matrix getActivation(Matrix z){
        Matrix tempZ2 = new DenseMatrix(z,true);
        tempZ2.scale(2);
        return MTJOpExt.divideExtend(MTJOpExt.minusExtend(MTJOpExt.powExtend(MTJCreateExt.single(Math.E), tempZ2), MTJCreateExt.single(1)), MTJOpExt.plusExtend(MTJOpExt.powExtend(MTJCreateExt.single(Math.E), tempZ2), MTJCreateExt.single(1)));
    }
    
    // returns f'(z) where f is the tanh function
    public Matrix getRawDerivative(Matrix z){
        return MTJOpExt.minusExtend(MTJCreateExt.single(1), MTJOpExt.powExtend(getActivation(z), MTJCreateExt.single(2)));
    }
    
    // returns g'(a)
    public Matrix getActivationDerivative(Matrix a){
        return MTJOpExt.minusExtend(MTJCreateExt.single(1), MTJOpExt.powExtend(a, MTJCreateExt.single(2)));
    }
    
    // Normalizes all features in a with the mean values in mu and the standard deviations in sig
    // normalizes for tanh with values between [-1,1]
    public static Matrix tanhFeatureNormalize(Matrix a, Matrix mu, Matrix sig){
        return MTJOpExt.minusExtend(MTJOpExt.timesExtend(MTJMathExt.featureNormalize(a, mu, sig), MTJCreateExt.single(2)), MTJCreateExt.single(1));
    }
}