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
    
    // Reshap vector a into an nr*nc matrix using rows rs to rf of a
    public static Matrix reshape(Matrix a, int rs, int rf, int nr, int nc){
        double[][] retArr = new double[nr][nc];
        int rAt = 0;
        int cAt = 0;
        for(int r = rs; r <= rf; r++){
            retArr[rAt][cAt] = a.get(r, 0);
            cAt++;
            if(cAt >= nc){
                cAt = 0;
                rAt++;
            }
        }
        return new DenseMatrix(retArr);
    }
    
    // return a vector with all Matricies in a in vector form concatenated to each other
    public static Matrix unroll(Matrix[] a){
        Matrix ret = toVector(a[0]);
        for(int i = 1; i < a.length; i++){
            ret = MTJConcat.concat(ret, toVector(a[i]), 2);
        }
        return ret;
    }
    
    /* maps the two input features (n*1 Matricies) to quadratic features.
     * Returns a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1.*X2, X1.*X2.^2, etc..*/
    public static Matrix mapFeature(Matrix X1, Matrix X2, int deg){
        if(X1.numColumns() != 1 || X2.numColumns() != 1 || X1.numRows() != X2.numRows())
            throw new IllegalArgumentException("X1 and X2 must be the same size");
        int colNums = 0;
        for(int i = 1; i <= deg+1; i++)
            colNums+=i;
        double[][] retArr = new double[X1.numRows()][colNums-1];
        int colAt = 0;
        for(int i = 1; i <= deg; i++){
            for(int j = 0; j <= i; j++){
                Matrix X1Pow = MTJOpExt.powExtend(X1, single(i-j));
                Matrix X2Pow = MTJOpExt.powExtend(X2, single(j));
                double[][] vec = GeneralFunctions.getMatrixArray(MTJOpExt.timesExtend(X1Pow, X2Pow));
                for(int r = 0; r < X1.numRows(); r++){
                    retArr[r][colAt] = vec[r][0];
                }
                colAt++;
            }
        }
        
        return MTJConcat.concat(Ones(X1.numRows(), 1), new DenseMatrix(retArr), 1);
    }
    
    // cuts a new double array from d from rows rs-rf and columns cs-cf
    public static double[][] splitDouble(double[][] d, int rs, int rf, int cs, int cf){
        if(rf == -1)
            rf = d.length-1;
        if(cf == -1)
            cf = d[0].length-1;
        double[][] ret = new double[rf-rs+1][cf-cs+1];
        int curR = 0;
        for(int r = rs; r <= rf; r++){
            int curC = 0;
            for(int c = cs; c <= cf; c++){
                ret[curR][curC] = d[r][c];
                curC++;
            }
            curR++;
        }
        return ret;
    }
    
    // cuts a new matrix from a from rows rs-rf and columns cs-cf
    public static Matrix splitMatrix(Matrix a, int rs, int rf, int cs, int cf){
        if(rf == -1)
            rf = a.numRows()-1;
        if(cf == -1)
            cf = a.numColumns()-1;
        double[][] ret = new double[rf-rs+1][cf-cs+1];
        int curR = 0;
        for(int r = rs; r <= rf; r++){
            int curC = 0;
            for(int c = cs; c <= cf; c++){
                ret[curR][curC] = a.get(r,c);
                curC++;
            }
            curR++;
        }
        return new DenseMatrix(ret);
    }

}