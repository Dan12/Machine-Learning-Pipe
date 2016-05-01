package nn;

import no.uib.cipr.matrix.*;

import java.util.Random;

class WeightMatrix{
    
    private DenseMatrix matrix;
    
    public WeightMatrix(int m, int n){
        this.matrix = (DenseMatrix) Matrices.random(m, n);
        double epsilon = Math.pow(n, 0.5);
        MTJOpExt.minusExtend(MTJOpExt.timesExtend(this.matrix, MTJCreateExt.single(2*epsilon)), MTJCreateExt.single(epsilon));
    }
    
    public DenseMatrix getMatrix(){
        
        return this.matrix;
    }
    
    public void setMatrix(DenseMatrix m){
        
        this.matrix = m;
    }
    
}