package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

// MTJ matrix creation extensions
public class MTJCreateExt {

    public MTJCreateExt(){}
    
    // returns m*n matrix of ones
    public static Matrix Ones(int m, int n){
        double[][] temp = new double[m][n];
        for(int r = 0; r < m; r++){
            for(int c = 0; c < n; c++){
                temp[r][c] = 1;
            }
        }
        return new DenseMatrix(temp);
    }
    
    // returns m*n matrix with all values set to con
    public static Matrix Const(int m, int n, double con){
        double[][] temp = new double[m][n];
        for(int r = 0; r < m; r++){
            for(int c = 0; c < n; c++){
                temp[r][c] = con;
            }
        }
        return new DenseMatrix(temp);
    }
    
    // returns m*n matrix of zeros
    public static Matrix Zeros(int m, int n){
        return new DenseMatrix(new double[m][n]);
    }
    
    // returns 1*x matrix of values from s-f at intervals i
    public static Matrix Range(int s, int i, int f){
        double[][] retArr = new double[1][(f-s+1)/i];
        int ind = 0;
        for(int v = s; v <= f; v+=i){
            retArr[0][ind] = v;
            ind++;
        }
        return new DenseMatrix(retArr);
    }
    
    // returns 1*1 matrix of d
    public static Matrix single(double d){
        return new DenseMatrix(new double[][]{{d}});
    }
    
    // returns x*1 matrix representation of a by stacking columns under each other
    public static Matrix toVector(Matrix a){
        double[][] retArr = new double[a.numRows()*a.numColumns()][1];
        int retAt = 0;
        for(int r = 0; r < a.numRows(); r++){
            for(int c = 0; c < a.numColumns(); c++){
                retArr[retAt][0] = a.get(r, c);
                retAt++;
            }
        }
        return new DenseMatrix(retArr);
    }
    
    // returns a 1*x or x*1 matrix with the max along the specified dim (1-rows (->), 2-cols (v))
    public static Matrix max(Matrix a, int dim){
        if(dim == 1){
            double[][] temp = new double[a.numRows()][2];
            for(int r = 0; r < a.numRows(); r++){
                double max = a.get(r,0);
                int index = 0;
                for(int c = 1; c < a.numColumns(); c++){
                    if(a.get(r, c) > max){
                        max = a.get(r,c);
                        index = c;
                    }
                }
                temp[r][0] = max;
                temp[r][1] = index;
            }
            return new DenseMatrix(temp);
        }
        else{
            double[][] temp = new double[2][a.numColumns()];
            for(int c = 0; c < a.numColumns(); c++){
                double max = a.get(0,c);
                int index = 0;
                for(int r = 1; r < a.numRows(); r++){
                    if(a.get(r, c) > max){
                        max = a.get(r,c);
                        index = r;
                    }
                }
                temp[0][c] = max;
                temp[1][c] = index;
            }
            return new DenseMatrix(temp);
        }
    }

}