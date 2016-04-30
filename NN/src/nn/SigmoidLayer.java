package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

class SigmoidLayer extends AbstractLayer{
    
    private int size;
    private Matrix activations;
    private Matrix gradients;
    
    private AbstractLayer outputLayer;
    private AbstractLayer inputLayer;
    private WeightMatrix weightMatrix;
    
    private boolean bias;
    
    public SigmoidLayer(int size, boolean bias){
        this.size = size;
        
        this.bias = bias;
    }
    
    public void test(){
        System.out.println("Hello");
    }
    
    // called with activations from previous layer, calculates activations, and sends them to the next layer
    public void feedForward(Matrix activations, Matrix target){
        int m = activations.numRows();
        int n = this.size;
        
        // calculate activations using the sigmoid function
        Matrix a = MTJMathExt.sigmoid(activations.transBmult(this.weightMatrix.getMatrix(), new DenseMatrix(m, n)));
        
        if(this.bias)
            this.activations = MTJConcat.concat(MTJCreateExt.Ones(m,1),a,1);
            
        // if we are not at the output, keep going
        if(this.outputLayer != null)
            this.outputLayer.feedForward(this.activations, target);
        // if this is the output, start going backwards
        else
            this.inputLayer.backProp(MTJOpExt.minusExtend(this.activations, target));
    }
    
    // called with errors from previous layer, calculated new errors and sends those back
    public void backProp(Matrix errors){
        // if the upper layer had a bias, cut that column out of the error term
        if(this.outputLayer.hasBias())
            this.gradients = MTJCreateExt.splitMatrix(errors, 0, -1, 1, -1).transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        else
            this.gradients = errors.transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
            
        // call backprop on the previous layer with this layer's error
        this.inputLayer.backProp(MTJOpExt.timesExtend(errors.mult(this.weightMatrix.getMatrix(), new DenseMatrix(this.activations.numRows(), this.activations.numColumns())), MTJMathExt.sigmoidGradientA(this.activations)));
    }
    
    // connect this layer to the next layer, called with next layer so return this layer with returnPipe
    public void pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.getSize()+ (this.bias ? 1 : 0), this.outputLayer.getSize());
        this.outputLayer.returnPipe(this);
    }
    
    // called by node with this input piped to it;
    public void returnPipe(AbstractLayer previousLayer){
        this.inputLayer = previousLayer;
    }
    
    public int getSize(){
        return this.size;
    }
    
    public boolean hasBias(){
        return this.bias;
    }
    
}