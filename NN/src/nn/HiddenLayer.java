package nn;

import no.uib.cipr.matrix.*;

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
    private AbstractGradientUpdate weightUpdate;
    
    public HiddenLayer(int size, boolean bias, AbstractActivationFunction activationFunc, AbstractRegularization weightReg, AbstractGradientUpdate update){
        this.size = size;
        
        this.bias = bias;
        
        this.activationFunction = activationFunc;
        
        this.weightRegularization = weightReg;
        
        this.weightUpdate = update;
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

        return this.outputLayer.feedForward(this.activations.transBmult(this.weightMatrix.getMatrix(), new DenseMatrix(m, this.outputLayer.getSize())), target) + this.weightRegularization.regularizeCost(this.weightMatrix.getMatrix(), this.bias)*(1.0/m);
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
            this.gradients = errors.transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        
        //System.out.println(this.gradients);    
        // add regularization (if bias, first column will be all 0s)
        this.gradients.add(this.weightRegularization.regularizeGradient(this.weightMatrix.getMatrix(), this.bias));
        //System.out.println(this.gradients);
        // divide the gradients by number of training cases
        this.gradients.scale(1.0/m);
            
        // call backprop on the previous layer with this layer's error ((error*weights).*activationGradient)
        Matrix unrolledGradients = this.inputLayer.backProp(MTJOpExt.timesExtend(errors.mult(this.weightMatrix.getMatrix(), new DenseMatrix(this.activations.numRows(), this.activations.numColumns())), this.activationFunction.getActivationDerivative(this.activations)));
        
        // return the unrolled input gradients concatenated with this layers gradients
        return MTJConcat.concat(MTJCreateExt.toVector(this.gradients), unrolledGradients, 2);
    }
    
    public void applyGradients(){
        
        this.weightMatrix.setMatrix((DenseMatrix) this.weightUpdate.gradientUpdate(this.weightMatrix.getMatrix(), this.gradients));
        
        this.inputLayer.applyGradients();
    }
    
    // connect this layer to the next layer, called with next layer so return this layer with returnPipe
    public AbstractLayer pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.outputLayer.getSize(), this.getSize() + (this.bias ? 1 : 0));
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
    
    public AbstractLayer getInputLayer(){
        return this.inputLayer;
    }
    
    public Matrix getActivations(){
        return this.activations;
    }
    
    public void setGradients(Matrix gradients){
        int rowFinish = this.gradients.numRows()*this.gradients.numColumns();
        
        this.gradients = MTJCreateExt.reshape(gradients, 0, rowFinish-1, this.gradients.numRows(), this.gradients.numColumns());
        
        this.inputLayer.setGradients(MTJCreateExt.reshape(gradients, rowFinish, gradients.numRows()-1, gradients.numRows()-rowFinish, 1));
    }
    
    public void setWeights(Matrix weights){
        int rowFinish = this.weightMatrix.getMatrix().numRows()*this.weightMatrix.getMatrix().numColumns();
        
        this.weightMatrix.setMatrix((DenseMatrix) MTJCreateExt.reshape(weights, 0, rowFinish-1, this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        
        this.inputLayer.setWeights(MTJCreateExt.reshape(weights, rowFinish, weights.numRows()-1, weights.numRows()-rowFinish, 1));
    }
    
    public Matrix getWeights(){
        return MTJConcat.concat(MTJCreateExt.toVector(this.weightMatrix.getMatrix()), this.inputLayer.getWeights(), 2);
    }
    
}