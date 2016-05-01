package nn;

import no.uib.cipr.matrix.*;

class InputLayer extends AbstractLayer{
    
    private int size;
    private Matrix activations;
    private Matrix gradients;
    
    private AbstractLayer outputLayer;
    private WeightMatrix weightMatrix;
    
    private boolean bias;
    
    private AbstractRegularization weightRegularization;
    private AbstractGradientUpdate weightUpdate;
    
    public InputLayer(int size, boolean bias, AbstractRegularization weightReg, AbstractGradientUpdate update){
        this.size = size;
        
        this.bias = bias;
        
        this.weightRegularization = weightReg;
        
        this.weightUpdate = update;
    }
    
    // called with information from previous layer, calculates activations, and sends them to the next layer
    public double feedForward(Matrix input, Matrix target){
        // number of training cases
        int m = input.numRows();
        
        if(this.bias)
            this.activations = MTJConcat.concat(MTJCreateExt.Ones(m,1), input, 1);
        else
            this.activations = input;
        
        // calculate z in the forward pass
        return this.outputLayer.feedForward(this.activations.transBmult(this.weightMatrix.getMatrix(), new DenseMatrix(m, this.outputLayer.getSize())), target) + this.weightRegularization.regularizeCost(this.weightMatrix.getMatrix(), this.bias)*(1.0/m);
    }
    
    // called with errors from previous layer, calculated new errors and sends those back
    public Matrix backProp(Matrix errors){
        // get the number of test cases
        int m = this.activations.numRows();
        
        // if the upper layer had a bias, cut that column out of the error term
        // then use the error matrix to calculate the weight gradients (error'*activations)
        if(this.outputLayer.hasBias())
            this.gradients = MTJCreateExt.splitMatrix(errors, 0, -1, 1, -1).transAmult((DenseMatrix) this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        else
            this.gradients = errors.transAmult(this.activations, new DenseMatrix(this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
        
        // regularize
        this.gradients.add(this.weightRegularization.regularizeGradient(this.weightMatrix.getMatrix(), this.bias));
        
        this.gradients.scale(1.0/m);
        
        return MTJCreateExt.toVector(this.gradients);
    }
    
    public void applyGradients(){
        
        this.weightMatrix.setMatrix((DenseMatrix) this.weightUpdate.gradientUpdate(this.weightMatrix.getMatrix(), this.gradients));
    }
    
    public AbstractLayer pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.outputLayer.getSize(), this.size + (this.bias ? 1 : 0));
        this.outputLayer.returnPipe(this);
        
        return this.outputLayer;
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
    
    public Matrix getActivations(){
        return this.activations;
    }
    
    public void setGradients(Matrix gradients){
        int rowFinish = this.gradients.numRows()*this.gradients.numColumns();
        
        this.gradients = MTJCreateExt.reshape(gradients, 0, rowFinish-1, this.gradients.numRows(), this.gradients.numColumns());
    }
    
    public void setWeights(Matrix weights){
        int rowFinish = this.weightMatrix.getMatrix().numRows()*this.weightMatrix.getMatrix().numColumns();
        
        this.weightMatrix.setMatrix((DenseMatrix) MTJCreateExt.reshape(weights, 0, rowFinish-1, this.weightMatrix.getMatrix().numRows(), this.weightMatrix.getMatrix().numColumns()));
    }
    
    public Matrix getWeights(){
        return MTJCreateExt.toVector(this.weightMatrix.getMatrix());
    }
}