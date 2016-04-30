package nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public abstract class AbstractActivationFunction{
    
    // returns f(z) where f is the activation function
    abstract Matrix getActivation(Matrix z);
    
    // returns f'(z) where f is the activation function
    abstract Matrix getDerivative(Matrix z);
    
    // returns g'(a) where a is a special matrix and g is a special derivative function
    // for sigmoid, a is activations and g is simplified derivative
    abstract Matrix getSpecialDerivative(Matrix a);
}