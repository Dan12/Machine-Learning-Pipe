package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class MTJConcat{
    
    public MTJConcat(){}
    
    // concat along dim (1-rows, 2-cols)
    public static Matrix concat(Matrix a, Matrix b, int dim){
        if(dim == 1){
            if(a.numRows()!= b.numRows())
                throw new IllegalArgumentException("All rows must have the same length.");
            int newN = a.numColumns()+b.numColumns();
            double[][] temp = new double[a.numRows()][newN];
            for(int r = 0; r < a.numRows(); r++){
                int curC = 0;
                for(int c = 0; c < newN; c++){
                    if(curC < a.numColumns())
                        temp[r][c] = a.get(r,curC);
                    else
                        temp[r][c] = b.get(r,curC-a.numColumns());
                    curC++;
                }
            }
            return new DenseMatrix(temp);
        }
        else{
            if(a.numColumns()!= b.numColumns())
                throw new IllegalArgumentException("All columns must have the same length.");
            int newM = a.numRows()+b.numRows();
            double[][] temp = new double[newM][a.numColumns()];
            int curR = 0;
            for(int r = 0; r < newM; r++){
                for(int c = 0; c < a.numColumns(); c++){
                    if(curR < a.numRows())
                        temp[r][c] = a.get(curR,c);
                    else
                        temp[r][c] = b.get(curR-a.numRows(),c);
                }
                curR++;
            }
            return new DenseMatrix(temp);
        }
    }
}