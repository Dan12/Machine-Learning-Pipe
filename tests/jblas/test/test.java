import org.jblas.*;
import org.jblas.DoubleMatrix.*;
import org.jblas.MatrixFunctions.*;

class test{
    public static void main(String[] args){
    	double[][] data = new double[][]
                       {{ 1,  2,  3,   4,   5},
                        { 6,  7,  8,   9,  10},
                        {11, 12, 13, 14, 15}};
    	DoubleMatrix matrix = new DoubleMatrix(data);
     
    	DoubleMatrix vector = new DoubleMatrix(new double[]{1, 1, 1, 1,1});
    	DoubleMatrix result = matrix.mmul(vector);
    	System.out.println(result.rows+"x"+result.columns+": "+result);
    }
}