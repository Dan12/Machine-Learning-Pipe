package nn;

import no.uib.cipr.matrix.*;

public class AlphaUpdate extends AbstractGradientUpdate{
    
    private double alpha;
    
    public AlphaUpdate(double a){
        this.alpha = a;
    }
    
    public Matrix gradientUpdate(Matrix weight, Matrix gradient){
        
        return MTJOpExt.minusExtend(weight, new DenseMatrix(gradient, true).scale(this.alpha));
    }
}