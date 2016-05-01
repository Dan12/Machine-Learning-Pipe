package nn;

import no.uib.cipr.matrix.*;

public class AutomatedLearningRate extends AbstractGradientUpdate{
    
    private double baseAlpha;
    private double alphaGrow;
    private double alphaShrink;
    private double alphaMin;
    private double alphaMax;
    
    private Matrix individualAlphas;
    private Matrix previousGradients;
    
    public AutomatedLearningRate(double base, double grow, double shrink, double min, double max){
        this.baseAlpha = base;
        this.alphaGrow = grow;
        this.alphaShrink = shrink;
        this.alphaMin = min;
        this.alphaMax = max;
    }
    
    public Matrix gradientUpdate(Matrix weight, Matrix gradient){
        
        if(this.individualAlphas == null)
            this.individualAlphas = MTJCreateExt.Const(gradient.numRows(), gradient.numColumns(), this.baseAlpha);
        else{
            Matrix signMatrix = MTJOpExt.timesExtend(gradient, this.previousGradients);
            Matrix positive = MTJOpExt.greaterThanExtend(signMatrix, 0).scale(this.alphaGrow);
            Matrix negative = MTJOpExt.lessThanExtend(signMatrix, 0).scale(this.alphaShrink);
            this.individualAlphas = MTJMathExt.min(MTJMathExt.max(MTJOpExt.specialZeroMultiplyRegular(negative, MTJOpExt.plusExtend(this.individualAlphas, positive)), this.alphaMax), this.alphaMin);
        }
        
        this.previousGradients = new DenseMatrix(gradient, true);
        
        return MTJOpExt.minusExtend(weight, MTJOpExt.timesExtend(gradient, this.individualAlphas));
    }
    
}