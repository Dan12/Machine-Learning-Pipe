package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

class InputLayer extends AbstractLayer{
    
    private int size;
    private DenseMatrix activations;
    private DenseMatrix gradients;
    
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
    public void feedForward(DenseMatrix activations){
        this.outputLayer.feedForward(activations);
    }
    
    // called with errors from previous layer, calculated new errors and sends those back
    public void backProp(){
        
    }
    
    public void pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.size, this.outputLayer.getSize());
        this.outputLayer.returnPipe(this);
    }
    
    public void returnPipe(AbstractLayer previousLayer){
        // nothing
        System.out.println("The input layer doesn't have a previous layer");
    }
    
    public int getSize(){
        return this.size;
    }
    
    // calculate the sigmoid of the matrix a
    private DenseMatrix sigmoid(DenseMatrix a){
        return null;
    }
}