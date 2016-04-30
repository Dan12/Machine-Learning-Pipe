import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Matrices;

class test{
    public static void main(String[] args){
        for(int i = 10; i <= 1000; i*=10){
            int r = i;
            int c = i;
            
        	DenseMatrix test1 = (DenseMatrix)Matrices.random(r,c);
        	DenseMatrix test2 = (DenseMatrix)Matrices.random(r,c);
        	DenseMatrix res = new DenseMatrix(r,c);
        	
        	long startTime = System.nanoTime();
        	test1.mult(test2, res);
            long endTime = System.nanoTime();
    
            long duration = (endTime - startTime);
        	
        	//System.out.println(res);
        	
        	System.out.println("I: "+i);
        	
        	System.out.println(duration+"ns");
        	
        	System.out.println(duration/(1000000000.0)+"s");
        	System.out.println("--------");
        }
    }
}