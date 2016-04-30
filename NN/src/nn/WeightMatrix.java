package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

class WeightMatrix{
    
    private DenseMatrix matrix;
    
    public WeightMatrix(DenseMatrix m){
        this.matrix = m;
    }
    
    public WeightMatrix(int m, int n){
        this.matrix = new DenseMatrix(m, n);
    }
    
    public DenseMatrix getMatrix(){
        
        return this.matrix;
    }
    
    public void setMatrix(DenseMatrix m){
        
        this.matrix = m;
    }
    
}