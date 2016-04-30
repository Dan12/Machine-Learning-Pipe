package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

class SigmoidLayer extends AbstractLayer{
    
    private int size;
    private DenseMatrix activations;
    private DenseMatrix gradients;
    
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
    public void feedForward(DenseMatrix activations){
        // 
        this.outputLayer.feedForward((DenseMatrix)
            activations.transBmult(
            weightMatrix.getMatrix(), 
            new DenseMatrix(activations.numRows(), this.size)));
    }
    
    // called with errors from previous layer, calculated new errors and sends those back
    public void backProp(){
        
    }
    
    // connect this layer to the next layer, called with next layer so return this layer with returnPipe
    public void pipe(AbstractLayer nextLayer){
        this.outputLayer = nextLayer;
        this.weightMatrix = new WeightMatrix(this.size, this.outputLayer.getSize());
        this.outputLayer.returnPipe(this);
    }
    
    // called by node with this input piped to it;
    public void returnPipe(AbstractLayer previousLayer){
        this.inputLayer = previousLayer;
    }
    
    public int getSize(){
        return this.size;
    }
}