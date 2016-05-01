package nn;

import no.uib.cipr.matrix.*;

public class OutputLayer extends HiddenLayer{
    
    private AbstractCostFunction costFunction;
    
    private Matrix activations;
    private AbstractActivationFunction activationFunction;
    
    private AbstractLayer inputLayer;
    
    public OutputLayer(int size, AbstractActivationFunction activationFunc, AbstractCostFunction costFunction){
        super(size, false, activationFunc, null, null);
        
        this.activationFunction = activationFunc;
        
        this.costFunction = costFunction;
    }
    
    @Override
    public double feedForward(Matrix z, Matrix target){
        int m = z.numRows();
        
        // calculate activations using the activation function
        this.activations = this.activationFunction.getActivation(z);
        
        return (1.0/m)*this.costFunction.getCost(this.activations, target);
    }
    
    @Override
    public Matrix backProp(Matrix target){
        return this.getInputLayer().backProp(this.costFunction.getError(this.activations, target, this.activationFunction));
    }
    
    @Override
    public void applyGradients(){
        this.getInputLayer().applyGradients();
    }
    
    @Override
    public Matrix getActivations(){
        return this.activations;
    }
    
    @Override
    public void setGradients(Matrix gradients){
        this.getInputLayer().setGradients(gradients);
    }
    
    @Override
    public void setWeights(Matrix weights){
        this.getInputLayer().setWeights(weights);
    }
    
    @Override 
    public Matrix getWeights(){
        return this.getInputLayer().getWeights();
    }
}