package nn;

import no.uib.cipr.matrix.*;

// General Functions
public class GeneralFunctions {

    public GeneralFunctions(){}
    
    // returns 2-dim double array of all values in a
    public static double[][] getMatrixArray(Matrix a){
        double[][] ret = new double[a.numRows()][a.numColumns()];
        for(int r = 0; r < ret.length; r++){
            for(int c = 0; c < ret[0].length; c++){
                ret[r][c] = a.get(r, c);
            }
        }
        return ret;
    }
    
    // returns string representation of d to highest prescision
    public static String doubleToString(double[][] d){
        String ret = "";
        for(int r = 0; r < d.length; r++){
            for(int c = 0; c < d[0].length; c++){
                ret+="  "+d[r][c];
            }
            ret+="\n";
        }
        return ret;
    }
    
    // returns string representation of matrix, unformated
    public static String matrixToString(Matrix a){
        return doubleToString(getMatrixArray(a));
    }

    // maps x in range in_min-in_max to range out_min-out_max
    public static double map(double x, double in_min, double in_max, double out_min, double out_max){
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }
}