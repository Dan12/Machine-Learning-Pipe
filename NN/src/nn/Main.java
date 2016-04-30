package nn;

public class Main{

    public static void main(String[] args){
        System.out.println("It works");
        
        TestClass.testmethod();
        
        InputLayer i = new InputLayer(5);
        
        i.test();
        
        i.pipe(new SigmoidLayer(5));
    }
    
}