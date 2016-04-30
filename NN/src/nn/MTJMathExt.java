package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

// MTJ math extensions
public class MTJMathExt{
    
    public MTJMathExt(){}
    
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