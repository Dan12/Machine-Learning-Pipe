package nn;

public class OutputLayer extends HiddenLayer{
    
    AbstractCostFunction costFunction;
    
    public OutputLayer(int size, boolean bias, AbstractActivationFunction activationFunc, AbstractRegularization weightReg, AbstractCostFunction costFunction){
        super(size, bias, activationFunc, weightReg);
        
        this.costFunction = costFunction;
    }
}