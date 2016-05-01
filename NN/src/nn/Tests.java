package nn;

import no.uib.cipr.matrix.*;

public class Tests{
    
    public Tests(){}
    
    public static void doAllTests(){
        testMTJExtensions();
        
        testBasicNN();
        
    }
    
    public static void testMTJExtensions(){
        DenseMatrix testData = new DenseMatrix(new double[][]{
            {1.0,2.0,0.1,0.4,4.2,-0.1,-0.5,0.001,0.005,0.0000005}
        });
        
        System.out.println((new TanhActivation()).getActivation(testData));
        System.out.println((new TanhActivation()).getActivationDerivative(testData));
        
    }
    
    public static void testBasicNN(){
        double lambda = 0.0001;
        int iters = 1000;
        double alpha = 1;
        double add = 0.03*alpha;
        double mult = 0.70;
        double min = 0.01;
        double max = 50;
        
        AbstractLayer inputLayer = new InputLayer(2, false, new L2Regularization(lambda), new AutomatedLearningRate(alpha,add,mult,min,max));
        
        AbstractLayer outputLayer = inputLayer
            .pipe(new HiddenLayer(2, false, new SigmoidActivation(), new L2Regularization(lambda), new AutomatedLearningRate(alpha,add,mult,min,max)))
            .pipe(new OutputLayer(1, new SigmoidActivation(), new CrossEntropyError()));
            
        DenseMatrix testData = new DenseMatrix(new double[][]{
            {0.0,0.0},
            {1.0,0.0},
            {0.0,1.0},
            {1.0,1.0}
        });
        
        DenseMatrix outputData = new DenseMatrix(new double[][]{
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        });
        
        for(int i = 0; i < iters; i++){
            System.out.println(inputLayer.feedForward(testData, outputData));
            outputLayer.backProp(outputData);
            outputLayer.applyGradients();
        }
        System.out.println(outputLayer.getActivations());
        
        Matrix gradients = outputLayer.backProp(outputData);
        outputLayer.setGradients(gradients);
        System.out.println(outputLayer.getWeights());
        //System.out.println(outputLayer.getActivations());
        //System.out.println(outputLayer.backProp(outputData));
    }
}
