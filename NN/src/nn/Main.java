package nn;

public class Main{

    public static void main(String[] args){
        System.out.println("Initializing the epicness");
        
        Tests.testMTJExtensions();
        
        AbstractLayer inputLayer = new InputLayer(5, false, new L2Regularization(0.1));
        
        AbstractLayer outputLayer = inputLayer
            .pipe(new HiddenLayer(5, false, new SigmoidActivation(), new L2Regularization(0.1)))
            .pipe(new OutputLayer(5, false, new SigmoidActivation(), new L2Regularization(0.1), new SquaredError()));
        
    }
    
}