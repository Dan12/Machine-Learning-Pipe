package nn;

import no.uib.cipr.matrix.*;

abstract class AbstractGradientUpdate{
    
    abstract Matrix gradientUpdate(Matrix weight, Matrix gradient);
}