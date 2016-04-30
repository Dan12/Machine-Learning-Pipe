package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

class InputLayer extends AbstractLayer{
    
    private int size;
    private Matrix activations;
    private Matrix gradients;
    
    private AbstractLayer outputLayer;
    private WeightMatrix weightMatrix;
    
    private boolean bias;
    
    public InputLayer(int size, boolean bias){
        this.size = size;
        
        this.bias = bias;
    }
    
    public void test(){
        System.out.println("Input with size "+this.size);
    }
    
    // called with information from previous layer, calculates activations, and sends them to the next layer
    public void feedForward(Matrix activations, Matrix target){
        // number of training cases
        int m = activations.numRows();
        
        if(this.bias)
            this.activations = MTJConcat.concat(MTJCreateExt.Ones(m,1), activations, 1);
        else
            this.activations = activations;
        
        this.outputLayer.feedForward(activations, target);
    }
    
    // called with errors from previous layer, calculated new errors and sends those back
    public void backProp(Matrix errors){
        
        // if the upper layer had a bias, cut that column out of the error term
        if(this.outputLayer.hasBias())
            this.gradients = MTJCreateExt.splitMatrix(errors, 0, -1, 1, -1).transAmult((DenseMatrix) this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        else
            this.gradients = errors.transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
            
        // call backprop on the previous layer with this layer's error
        // this.inputLayer.backProp(MTJOpExt.timesExtend(errors.mult(this.weightMatrix.getMatrix()), MTJMathExt.sigmoidGradientA(this.activations)));   
    }
    
    public void pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.size + (this.bias ? 1 : 0), this.outputLayer.getSize());
        this.outputLayer.returnPipe(this);
    }
    
    public void returnPipe(AbstractLayer previousLayer){
        // nothing
        System.out.println("The input layer doesn't have a previous layer");
    }
    
    public int getSize(){
        return this.size;
    }
    
    public boolean hasBias(){
        return this.bias;
    }
    
}