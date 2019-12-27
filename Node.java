import java.util.*;
import java.lang.Math;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient
    private double sumZ = 0.0; //sum of the z for 3 outputs
    private double z = 0.0;

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {   //Not an input or bias node
            // TODO: add code here
        	double g = 0;
        	//Computing z
        	
        	if (type == 2) {
        		computingZ();
        		g = Math.max(0, z);
        	}
        	else {
        		g = Math.exp(z)/sumZ;
        	}
        	outputValue = g;
        }
    }

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    //Calculate the delta value of a node.
    public void calculateDelta(double y, double deltaHelper) {
        if (type == 2 || type == 4)  {
            // TODO: add code here
        	if (type == 2) {
        		if (z > 0) delta = deltaHelper;
        		else delta = 0;
        	}
        	else {
        		delta = y - getOutput();
        	}
        }
    }


    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            // TODO: add code here
        	for (int i = 0; i < parents.size(); ++i) {
        		parents.get(i).weight += learningRate * parents.get(i).node.getOutput() * delta;	
        	}
        	
        }
    }
    
    public void updateZ (double sumZ) {
    	this.sumZ = sumZ;
    }
    
    public double computingZ () {
    	double z = 0;
    	for (int i = 0; i < parents.size(); ++i) {
    		z += parents.get(i).node.getOutput() * parents.get(i).weight;
    	}
    	this.z = z;
    	return z;
    }
    
    public double computingDeltaHelper(int idx) {
    	return delta * parents.get(idx).weight;
    }
    
}


