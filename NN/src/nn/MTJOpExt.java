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
            if(op == 0)
                result.add(b);
            else if(op == 1){
                Matrix invB = new DenseMatrix(b, true);
                invB.scale(-1);
                result.add(invB);
            }
            else{
                result = null;
                double[][] retArr = new double[a.numRows()][a.numColumns()];
                for(int r = 0; r < a.numRows(); r++){
                    for(int c = 0; c < a.numColumns(); c++){
                        if(op == 2)
                            retArr[r][c] = a.get(r, c)*b.get(r, c);
                        else if(op == 3)
                            retArr[r][c] = a.get(r, c)/b.get(r, c);
                        else if(op == 4)
                            retArr[r][c] = Math.pow(a.get(r, c),b.get(r, c));
                        else if(op == 5)
                            retArr[r][c] = Math.log(a.get(r, c));
                        else if(op == 6){
                            if(a.get(r,c) == b.get(r,c))
                                retArr[r][c] = 1;
                            else
                                retArr[r][c] = 0;
                        }
                        else if(op == 7)
                            retArr[r][c] = Math.round(a.get(r,c));
                        else if(op == 8)
                            retArr[r][c] = ((int) a.get(r, c))%((int)b.get(r, c));
                    }
                }
                result = new DenseMatrix(retArr);
            }
            return result;
        }
        // if the columns are equal
        else if(a.numColumns() == b.numColumns()){
            if(a.numRows() == 1){
                double[][] retArr = new double[b.numRows()][b.numColumns()];
                for(int r = 0; r < b.numRows(); r++){
                    for(int c = 0; c < a.numColumns(); c++){
                        switch(op){
                            case 0:
                                retArr[r][c] = a.get(0, c)+b.get(r,c);
                                break;
                            case 1:
                                retArr[r][c] = a.get(0, c)-b.get(r,c);
                                break;
                            case 2:
                                retArr[r][c] = a.get(0, c)*b.get(r,c);
                                break;
                            case 3:
                                retArr[r][c] = a.get(0, c)/b.get(r,c);
                                break;
                            case 4:
                                retArr[r][c] = Math.pow(a.get(0, c),b.get(r,c));
                                break;
                            case 6:
                                if(a.get(0,c) == b.get(r,c))
                                    retArr[r][c] = 1;
                                else
                                    retArr[r][c] = 0;
                                break;
                            case 8:
                                retArr[r][c] = ((int) a.get(0, c))%((int)b.get(r, c));
                                break;
                        }
                    }
                }
                return new DenseMatrix(retArr);
            }
            else if(b.numRows() == 1){
                double[][] retArr = new double[a.numRows()][a.numColumns()];
                for(int r = 0; r < a.numRows(); r++){
                    for(int c = 0; c < a.numColumns(); c++){
                        switch(op){
                            case 0:
                                retArr[r][c] = a.get(r, c)+b.get(0,c);
                                break;
                            case 1:
                                retArr[r][c] = a.get(r, c)-b.get(0,c);
                                break;
                            case 2:
                                retArr[r][c] = a.get(r, c)*b.get(0,c);
                                break;
                            case 3:
                                retArr[r][c] = a.get(r, c)/b.get(0,c);
                                break;
                            case 4:
                                retArr[r][c] = Math.pow(a.get(r, c),b.get(0,c));
                                break;
                            case 6:
                                if(a.get(r,c) == b.get(0,c))
                                    retArr[r][c] = 1;
                                else
                                    retArr[r][c] = 0;
                                break;
                            case 8:
                                retArr[r][c] = ((int) a.get(r, c))%((int)b.get(0, c));
                                break;
                        }
                    }
                }
                return new DenseMatrix(retArr);
            }
            else{
                throw new IllegalArgumentException("No Good Arguments.");
            }
        }
        else if(a.numRows() == b.numRows()){
            if(a.numColumns() == 1){
                double[][] retArr = new double[b.numRows()][b.numColumns()];
                for(int r = 0; r < a.numRows(); r++){
                    for(int c = 0; c < b.numColumns(); c++){
                        switch(op){
                            case 0:
                                retArr[r][c] = a.get(r, 0)+b.get(r,c);
                                break;
                            case 1:
                                retArr[r][c] = a.get(r, 0)-b.get(r,c);
                                break;
                            case 2:
                                retArr[r][c] = a.get(r, 0)*b.get(r,c);
                                break;
                            case 3:
                                retArr[r][c] = a.get(r, 0)/b.get(r,c);
                                break;
                            case 4:
                                retArr[r][c] = Math.pow(a.get(r, 0),b.get(r,c));
                                break;
                            case 6:
                                if(a.get(r,0) == b.get(r,c))
                                    retArr[r][c] = 1;
                                else
                                    retArr[r][c] = 0;
                                break;
                            case 8:
                                retArr[r][c] = ((int) a.get(r, 0))%((int)b.get(r, c));
                                break;
                        }
                    }
                }
                return new DenseMatrix(retArr);
            }
            else if(b.numColumns() == 1){
                double[][] retArr = new double[a.numRows()][a.numColumns()];
                for(int r = 0; r < a.numRows(); r++){
                    for(int c = 0; c < a.numColumns(); c++){
                        switch(op){
                            case 0:
                                retArr[r][c] = a.get(r, c)+b.get(r,0);
                                break;
                            case 1:
                                retArr[r][c] = a.get(r, c)-b.get(r,0);
                                break;
                            case 2:
                                retArr[r][c] = a.get(r, c)*b.get(r,0);
                                break;
                            case 3:
                                retArr[r][c] = a.get(r, c)/b.get(r,0);
                                break;
                            case 4:
                                retArr[r][c] = Math.pow(a.get(r, c),b.get(r,0));
                                break;
                            case 6:
                                if(a.get(r,c) == b.get(r,0))
                                    retArr[r][c] = 1;
                                else
                                    retArr[r][c] = 0;
                                break;
                            case 8:
                                retArr[r][c] = ((int) a.get(r, c))%((int)b.get(r, 0));
                                break;
                        }
                    }
                }
                return new DenseMatrix(retArr);
            }
            else{
                throw new IllegalArgumentException("No Good Arguments.");
            }
        }
        else if(a.numRows() == 1 && a.numColumns() == 1){
            double[][] retArr = new double[b.numRows()][b.numColumns()];
            for(int r = 0; r < b.numRows(); r++){
                for(int c = 0; c < b.numColumns(); c++){
                    switch(op){
                        case 0:
                            retArr[r][c] = a.get(0, 0)+b.get(r,c);
                            break;
                        case 1:
                            retArr[r][c] = a.get(0, 0)-b.get(r,c);
                            break;
                        case 2:
                            retArr[r][c] = a.get(0, 0)*b.get(r,c);
                            break;
                        case 3:
                            retArr[r][c] = a.get(0, 0)/b.get(r,c);
                            break;
                        case 4:
                            retArr[r][c] = Math.pow(a.get(0, 0),b.get(r,c));
                            break;
                        case 6:
                            if(a.get(0,0) == b.get(r,c))
                                retArr[r][c] = 1;
                            else
                                retArr[r][c] = 0;
                            break;
                        case 8:
                            retArr[r][c] = ((int) a.get(0, 0))%((int)b.get(r, c));
                            break;
                    }
                }
            }
            return new DenseMatrix(retArr);
        }
        else if(b.numRows() == 1 && b.numColumns() == 1){
            double[][] retArr = new double[a.numRows()][a.numColumns()];
            for(int r = 0; r < a.numRows(); r++){
                for(int c = 0; c < a.numColumns(); c++){
                    switch(op){
                        case 0:
                            retArr[r][c] = a.get(r, c)+b.get(0,0);
                            break;
                        case 1:
                            retArr[r][c] = a.get(r, c)-b.get(0,0);
                            break;
                        case 2:
                            retArr[r][c] = a.get(r, c)*b.get(0,0);
                            break;
                        case 3:
                            retArr[r][c] = a.get(r, c)/b.get(0,0);
                            break;
                        case 4:
                            retArr[r][c] = Math.pow(a.get(r, c),b.get(0,0));
                            break;
                        case 6:
                            if(a.get(r,c) == b.get(0,0))
                                retArr[r][c] = 1;
                            else
                                retArr[r][c] = 0;
                            break;
                        case 8:
                            retArr[r][c] = ((int) a.get(r, c))%((int)b.get(0, 0));
                            break;
                    }
                }
            }
            return new DenseMatrix(retArr);
        }
        else
            throw new IllegalArgumentException("No Good Arguments.");
    }
    
    private double opSwitch(int op, DenseMatrix a, DenseMatrix b, ar, ac, br, bc){
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
                break;
            // division
            case 3:
                return a.get(ar, ac)/b.get(br, bc);
            // power
            case 4:
                return Math.pow(a.get(ar, ac),b.get(br, bc));
                break;
            // natural log of first matrix
            case 5:
                return Math.log(a.get(ar, ac));
            // equals
            case 6:
                return a.get(ar,ac) == b.get(br, bc) ? 1 : 0;
            // round first matrix
            case 7
                return Math.round(a.get(ar, ac));
            // modulo
            case 8
                return ((int) a.get(ar, ac))%((int) b.get(br, bc));
        }
        
    }
    
}