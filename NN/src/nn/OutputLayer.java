package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class OutputLayer extends HiddenLayer{
    
    private AbstractCostFunction costFunction;
    
    private Matrix activations;
    private AbstractActivationFunction activationFunction;
    
    private AbstractLayer inputLayer;
    
    public OutputLayer(int size, AbstractActivationFunction activationFunc, AbstractRegularization weightReg, AbstractCostFunction costFunction){
        super(size, false, activationFunc, weightReg);
        
        this.costFunction = costFunction;
    }
    
    @Override
    public double feedForward(Matrix z, Matrix target){
        
        int m = z.numRows();
        
        // calculate activations using the activation function
        this.activations = this.activationFunction.getActivation(z);
        
        return this.costFunction.getCost(this.activations, target);
    }
    
    @Override
    public Matrix backProp(Matrix target){
        
        return this.inputLayer.backProp(this.costFunction.getError(this.activations, target));
    }
}