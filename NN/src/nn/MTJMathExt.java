package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

// MTJ math extensions
public class MTJMathExt{
    
    public MTJMathExt(){}
    
    // return a matrix with the mean along specified dimension (1-rows, get x*1 matrix; 2-cols, get 1*x matrix)
    public static Matrix mean(Matrix a, int dim){
        if(dim == 1){
            double[][] retArr = GenFunc.getMatrixArray(sum(a, 1));
            for(int r = 0; r < a.numRows(); r++){
                retArr[r][0] = retArr[r][0]/a.numColumns();
            }
            return new DenseMatrix(retArr);
        }
        else{
            double[][] retArr = GenFunc.getMatrixArray(sum(a, 2));
            for(int c = 0; c < a.numColumns(); c++){
                retArr[0][c] = retArr[0][c]/a.numRows();
            }
            return new DenseMatrix(retArr);
        }
    }
    
    // return a matrix with the standard deviation along specified dimension (1-rows, get x*1 matrix; 2-cols, get 1*x matrix)
    public static Matrix std(Matrix a, int dim){
        Matrix temp = sum(powExtend(minusExtend(a, mean(a, dim)),new DenseMatrix(new double[][]{{2}})),dim);
        double divisor = 0;
        if(dim == 1)
            divisor = a.numColumns();
        else
            divisor = a.numRows();
        temp.scale(((double)1)/divisor);
        return powExtend(temp, new DenseMatrix(new double[][]{{0.5}}));
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