package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

// the class should keep track of the input and output layers
// the class should keep track of activations

public abstract class AbstractLayer{

    // called with information from previous layer, calculates activations, and sends them to the next layer
    abstract double feedForward(Matrix z, Matrix target);
    
    // called with errors from previous layer, calculated new errors and sends those back
    abstract Matrix backProp(Matrix target);
    
    // apply gradients
    abstract void applyGradients();
    
    // connect this layer with the next layer
    // have to call return pipe on next layer to create doubly linked nodes
    abstract AbstractLayer pipe(AbstractLayer nextLayer);
    
    // connect this layer with the previous layer
    abstract void returnPipe(AbstractLayer previousLayer);
    
    // return layer size
    abstract int getSize();
    
    abstract boolean hasBias();
}