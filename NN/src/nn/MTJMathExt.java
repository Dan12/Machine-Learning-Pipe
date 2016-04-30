package nn;

import java.math.BigDecimal;
import java.math.RoundingMode;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

// MTJ math extensions
public class MTJMathExt{
    
    public MTJMathExt(){}
    
    // BigDecimal Precision
    private static final int precision = 100;
    
    // Calculate sigmoid of all values in z
    public static Matrix sigmoid(Matrix z){
        Matrix tempZ = new DenseMatrix(z,true);
        return MTJOpExt.divideExtend(MTJCreateExt.single(1), MTJOpExt.plusExtend(MTJCreateExt.single(1), MTJOpExt.powExtend(MTJCreateExt.single(Math.E), tempZ.scale(-1))));
    }
    
    // Calculate derivative of sigmoid for all values in z
    public static Matrix sigmoidGradient(Matrix z){
        return MTJOpExt.timesExtend(sigmoidEx(z), invSigmoidEx(z));
    }
    
    // Calculate sigmoid with BigDeciaml, high precision
    public static Matrix sigmoidEx(Matrix z){
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
    public static Matrix invSigmoidEx(Matrix z){
        double[][] retArr = new double[z.numRows()][z.numColumns()];
        for(int r = 0; r < z.numRows(); r++){
            for(int c = 0; c < z.numColumns(); c++){
                BigDecimal exp = new BigDecimal(Math.exp(-z.get(r, c)));
                retArr[r][c] = (new BigDecimal(1).subtract(new BigDecimal(1).divide(new BigDecimal(1).add(exp), precision, RoundingMode.HALF_UP))).doubleValue();
            }
        }
        return new DenseMatrix(retArr);
    }
    
    // return a matrix with the mean along specified dimension (1-rows, get x*1 matrix; 2-cols, get 1*x matrix)
    public static Matrix mean(Matrix a, int dim){
        if(dim == 1){
            double[][] retArr = GeneralFunctions.getMatrixArray(sum(a, 1));
            for(int r = 0; r < a.numRows(); r++){
                retArr[r][0] = retArr[r][0]/a.numColumns();
            }
            return new DenseMatrix(retArr);
        }
        else{
            double[][] retArr = GeneralFunctions.getMatrixArray(sum(a, 2));
            for(int c = 0; c < a.numColumns(); c++){
                retArr[0][c] = retArr[0][c]/a.numRows();
            }
            return new DenseMatrix(retArr);
        }
    }
    
    // return a matrix with the standard deviation along specified dimension (1-rows, get x*1 matrix; 2-cols, get 1*x matrix)
    public static Matrix std(Matrix a, int dim){
        // sum (a-mean(a))^2 along dim
        Matrix temp = sum(
            MTJOpExt.powExtend(
                MTJOpExt.minusExtend(a,mean(a, dim)),
                new DenseMatrix(new double[][]{{2}})),
            dim);
        double divisor = 0;
        if(dim == 1)
            divisor = a.numColumns();
        else
            divisor = a.numRows();
        temp.scale(((double)1)/divisor);
        return MTJOpExt.powExtend(temp, new DenseMatrix(new double[][]{{0.5}}));
    }
    
    // Normalizes all features in a with the mean values in mu and the standard deviations in sig
    public static Matrix featureNormalize(Matrix a, Matrix mu, Matrix sig){
        return MTJOpExt.divideExtend(MTJOpExt.minusExtend(a,mu), sig);
    }
    
    // return a matrix with the sum of all elements of the matrix along dim (1: ->, 2: v)
    public static Matrix sum(Matrix a, int dim){
        double[][] retArr = null;
        if(dim == 1){
            retArr = new double[a.numRows()][1];
            for(int r = 0; r < a.numRows(); r++){
                double sum = 0;
                for(int c = 0; c < a.numColumns(); c++){
                    sum+=a.get(r, c);
                }
                retArr[r] = new double[]{sum};
            }
            return new DenseMatrix(retArr);
        }
        else{
            retArr = new double[1][a.numColumns()];
            for(int c = 0; c < a.numColumns(); c++){
                double sum = 0;
                for(int r = 0; r < a.numRows(); r++){
                    sum+=a.get(r, c);
                }
                retArr[0][c] = sum;
            }
            return new DenseMatrix(retArr);
        }
    }
}