package nn;

import java.math.BigDecimal;
import java.math.RoundingMode;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class SigmoidActivation extends AbstractActivationFunction{
    
    public SigmoidActivation(){}
    
    // returns f(z) where f is the activation function
    public Matrix getActivation(Matrix z){
        Matrix tempZ = new DenseMatrix(z,true);
        return MTJOpExt.divideExtend(MTJCreateExt.single(1), MTJOpExt.plusExtend(MTJCreateExt.single(1), MTJOpExt.powExtend(MTJCreateExt.single(Math.E), tempZ.scale(-1))));
    }
    
    // returns f'(z) where f is the activation function
    public Matrix getDerivative(Matrix z){
        return MTJOpExt.timesExtend(sigmoidEx(z), invSigmoidEx(z));
    }
    
    // returns g'(a) where a is activations and g is simplified derivative
    public Matrix getSpecialDerivative(Matrix a){
        return MTJOpExt.timesExtend(a, MTJOpExt.minusExtend(MTJCreateExt.single(1.0),a));
    }
    
    // BigDecimal Precision
    private final int precision = 100;
    
    // Calculate sigmoid with BigDeciaml, high precision
    private Matrix sigmoidEx(Matrix z){
        double[][] retArr = new double[z.numRows()][z.numColumns()];
        for(int r = 0; r < z.numRows(); r++){
            for(int c = 0; c < z.numColumns(); c++){
                BigDecimal exp = new BigDecimal(Math.exp(-z.get(r, c)));
                retArr[r][c] = (new BigDecimal(1).divide(new BigDecimal(1).add(exp), precision, RoundingMode.HALF_UP)).doubleValue();
            }
        }
        return new DenseMatrix(retArr);
    }
    
    // Calculate 1-sigmoid with BigDecimal, high precision
    private Matrix invSigmoidEx(Matrix z){
        double[][] retArr = new double[z.numRows()][z.numColumns()];
        for(int r = 0; r < z.numRows(); r++){
            for(int c = 0; c < z.numColumns(); c++){
                BigDecimal exp = new BigDecimal(Math.exp(-z.get(r, c)));
                retArr[r][c] = (new BigDecimal(1).subtract(new BigDecimal(1).divide(new BigDecimal(1).add(exp), precision, RoundingMode.HALF_UP))).doubleValue();
            }
        }
        return new DenseMatrix(retArr);
    }
}