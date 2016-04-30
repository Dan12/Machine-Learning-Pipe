package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

// MTJ operation extensions
public class MTJOpExt{
    
    public MTJOpExt(){}
    
    public static Matrix plusExtend(Matrix a, Matrix b){
        return opExtend(a, b, 0);
    }
    
    public static Matrix minusExtend(Matrix a, Matrix b){
        return opExtend(a, b, 1);
    }
    
    public static Matrix timesExtend(Matrix a, Matrix b){
        return opExtend(a, b, 2);
    }
    
    public static Matrix divideExtend(Matrix a, Matrix b){
        return opExtend(a, b, 3);
    }
    
    public static Matrix powExtend(Matrix a, Matrix b){
        return opExtend(a,b,4);
    }
    
    public static Matrix logExtend(Matrix a){
        return opExtend(a, a, 5);
    }
    
    public static Matrix equalsExtend(Matrix a, Matrix b){
        return opExtend(a, b, 6);
    }
    
    public static Matrix roundExtend(Matrix a){
        return opExtend(a, a, 7);
    }
    
    public static Matrix moduloExtend(Matrix a, Matrix b){
        return opExtend(a, b, 8);
    }
    
    //Extends element-wise operations even if matricies do not have same dimensions
    //0-add, 1-right sub, 2-mult, 3-right div, 4-pow, 5-log, 6-equals(==), 7-round, 8-modulo
    private static Matrix opExtend(Matrix a, Matrix b, int op){
        
        
        // if the columns and rows are equal, perform the element wise operation
        if(a.numColumns() == b.numColumns() && a.numRows() == b.numRows()){
            DenseMatrix result = new DenseMatrix(a, true);
            // this is just a regular add or subtract
            if(op == 0)
                result.add(b);
            else if(op == 1){
                Matrix invB = new DenseMatrix(b, true);
                invB.scale(-1);
                result.add(invB);
            }
            
            // other than add/subtract, do elementwise
            else{
                result = null;
                double[][] retArr = new double[a.numRows()][a.numColumns()];
                for(int r = 0; r < a.numRows(); r++)
                    for(int c = 0; c < a.numColumns(); c++)
                        retArr[r][c] = opSwitch(op, a, b, r, c, r, c);
                result = new DenseMatrix(retArr);
            }
            return result;
        }
        
        
        // if the columns are equal
        else if(a.numColumns() == b.numColumns()){
            // a is the transpose vector (1*x)
            if(a.numRows() == 1){
                double[][] retArr = new double[b.numRows()][b.numColumns()];
                for(int r = 0; r < b.numRows(); r++)
                    for(int c = 0; c < a.numColumns(); c++)
                        retArr[r][c] = opSwitch(op, a, b, 0, c, r, c);
                return new DenseMatrix(retArr);
            }
            // b is the transpose vector (1*x)
            else if(b.numRows() == 1){
                double[][] retArr = new double[a.numRows()][a.numColumns()];
                for(int r = 0; r < a.numRows(); r++)
                    for(int c = 0; c < a.numColumns(); c++)
                        retArr[r][c] = opSwitch(op, a, b, r, c, 0, c);
                return new DenseMatrix(retArr);
            }
            else
                throw new IllegalArgumentException("No Good Arguments.");
        }
        
        
        // if the rows are equal
        else if(a.numRows() == b.numRows()){
            // a is the vector
            if(a.numColumns() == 1){
                double[][] retArr = new double[b.numRows()][b.numColumns()];
                for(int r = 0; r < a.numRows(); r++)
                    for(int c = 0; c < b.numColumns(); c++)
                        retArr[r][c] = opSwitch(op, a, b, r, 0, r, c);
                return new DenseMatrix(retArr);
            }
            // b is the vector
            else if(b.numColumns() == 1){
                double[][] retArr = new double[a.numRows()][a.numColumns()];
                for(int r = 0; r < a.numRows(); r++)
                    for(int c = 0; c < a.numColumns(); c++)
                        retArr[r][c] = opSwitch(op, a, b, r, c, r, 0);
                return new DenseMatrix(retArr);
            }
            else
                throw new IllegalArgumentException("No Good Arguments.");
        }
        
        
        // a is a 1*1
        else if(a.numRows() == 1 && a.numColumns() == 1){
            double[][] retArr = new double[b.numRows()][b.numColumns()];
            for(int r = 0; r < b.numRows(); r++)
                for(int c = 0; c < b.numColumns(); c++)
                    retArr[r][c] = opSwitch(op, a, b, 0, 0, r, c);
            return new DenseMatrix(retArr);
        }
        
        // b is a 1*1
        else if(b.numRows() == 1 && b.numColumns() == 1){
            double[][] retArr = new double[a.numRows()][a.numColumns()];
            for(int r = 0; r < a.numRows(); r++){
                for(int c = 0; c < a.numColumns(); c++){
                    retArr[r][c] = opSwitch(op, a, b, r, c, 0, 0);
                }
            }
            return new DenseMatrix(retArr);
        }
        else
            throw new IllegalArgumentException("No Good Arguments.");
            
    }
    
    private static double opSwitch(int op, Matrix a, Matrix b, int ar, int ac, int br, int bc){
        switch(op){
            // addition
            case 0:
                return a.get(ar, ac)+b.get(br, bc);
            // subtraction
            case 1:
                return a.get(ar, ac)-b.get(br, bc);
            // multiplication
            case 2:
                return a.get(ar, ac)*b.get(br, bc);
            // division
            case 3:
                return a.get(ar, ac)/b.get(br, bc);
            // power
            case 4:
                return Math.pow(a.get(ar, ac),b.get(br, bc));
            // natural log of first matrix
            case 5:
                return Math.log(a.get(ar, ac));
            // equals
            case 6:
                return a.get(ar,ac) == b.get(br, bc) ? 1 : 0;
            // round first matrix
            case 7:
                return Math.round(a.get(ar, ac));
            // modulo
            case 8:
                return ((int) a.get(ar, ac))%((int) b.get(br, bc));
        }
        
        throw new IllegalArgumentException("No Good Arguments in Operation Switch. No good Operation.");
        
    }
    
}