package nn;

public class Tests{
    
    public Tests(){}
    
    public static void doAllTests(){
        testMTJExtensions();
        
        testBasicNN();
        
    }
    
    public static void testMTJExtensions(){
        
        
    }
    
    public static void testBasicNN(){
        AbstractLayer inputLayer = new InputLayer(5, false, new L2Regularization(0.1));
        
        AbstractLayer outputLayer = inputLayer
            .pipe(new HiddenLayer(5, false, new SigmoidActivation(), new L2Regularization(0.1)))
            .pipe(new OutputLayer(5, new SigmoidActivation(), new L2Regularization(0.2), new SquaredError()));
    }
}
