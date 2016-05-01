package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

class HiddenLayer extends AbstractLayer{
    
    private int size;
    private Matrix activations;
    private Matrix gradients;
    
    private AbstractLayer outputLayer;
    private AbstractLayer inputLayer;
    private WeightMatrix weightMatrix;
    
    private boolean bias;
    
    private AbstractActivationFunction activationFunction;
    private AbstractRegularization weightRegularization;
    
    public HiddenLayer(int size, boolean bias, AbstractActivationFunction activationFunc, AbstractRegularization weightReg){
        this.size = size;
        
        this.bias = bias;
        
        this.activationFunction = activationFunc;
        
        this.weightRegularization = weightReg;
    }
    
    // called with raw values from previous layer, calculates activations, and sends them to the next layer
    public double feedForward(Matrix z, Matrix target){
        int m = z.numRows();
        
        // calculate activations using the activation function
        Matrix a = this.activationFunction.getActivation(z);
        
        if(this.bias)
            this.activations = MTJConcat.concat(MTJCreateExt.Ones(m,1),a,1);
        else
            this.activations = a;

        return this.outputLayer.feedForward(this.activations.transBmult(this.weightMatrix.getMatrix(), new DenseMatrix(m, this.outputLayer.getSize())), target) + this.weightRegularization.regularizeCost(this.weightMatrix.getMatrix(), this.bias);
    }
    
    // called with errors from previous layer, calculated new errors and sends those back
    public Matrix backProp(Matrix errors){
        // get the number of training cases
        int m = this.activations.numRows();
        
        // if the upper layer had a bias, cut that column out of the error term
        // then use the error matrix to calculate the weight gradients (error'*activations)
        if(this.outputLayer.hasBias())
            this.gradients = MTJCreateExt.splitMatrix(errors, 0, -1, 1, -1).transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        else
            this.gradients = errors.transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns() ));
            
        // add regularization (if bias, first column will be all 0s)
        this.gradients.add(this.weightRegularization.regularizeGradient(this.weightMatrix.getMatrix(), this.bias));
        
        // divide the gradients by number of training cases
        this.gradients.scale(1.0/m);
            
        // call backprop on the previous layer with this layer's error ((error*weights).*activationGradient)
        Matrix unrolledGradients = this.inputLayer.backProp(MTJOpExt.timesExtend(errors.mult(this.weightMatrix.getMatrix(), new DenseMatrix(this.activations.numRows(), this.activations.numColumns())), this.activationFunction.getSpecialDerivative(this.activations)));
        
        // return the unrolled input gradients concatenated with this layers gradients
        return MTJConcat.concat(unrolledGradients, MTJCreateExt.toVector(this.gradients), 2);
    }
    
    public void applyGradients(){
        
    }
    
    // connect this layer to the next layer, called with next layer so return this layer with returnPipe
    public AbstractLayer pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.getSize()+ (this.bias ? 1 : 0), this.outputLayer.getSize());
        this.outputLayer.returnPipe(this);
        
        return this.outputLayer;
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