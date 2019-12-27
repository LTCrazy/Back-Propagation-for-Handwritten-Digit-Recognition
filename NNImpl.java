import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        // TODO: add code here
//    	System.out.print("input nodes #:"+inputNodes.size()+"\n");
//    	System.out.print("attribute #:" +instance.attributes.size()+"\n");
    	for (int i = 0; i < inputNodes.size()-1; ++i) {
    		inputNodes.get(i).setInput(instance.attributes.get(i));
    	}
    	inputNodes.get(inputNodes.size()-1).setInput(1); //bias node
    	
    	for (int i = 0; i < hiddenNodes.size(); ++i) {
    		hiddenNodes.get(i).calculateOutput();
    	}
    	//Computing sumZ (denominator)
    	double sumZ = 0.0;
    	for (int i = 0; i < outputNodes.size(); ++i) {
    		sumZ += Math.exp(outputNodes.get(i).computingZ());    		
    	}
    	for (int i = 0; i < outputNodes.size(); ++i) {
    		outputNodes.get(i).updateZ(sumZ);
    		outputNodes.get(i).calculateOutput();
    	}
    	
    	double max = outputNodes.get(2).getOutput();
    	int idx = 2;
    	for (int i = 0; i < 2; ++i) {
    		if(outputNodes.get(i).getOutput() > max) {
    			max = outputNodes.get(i).getOutput();
    			idx = i;
    		}
    		
    	}
        return idx;
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        // TODO: add code here
    	//shuffle,predict,compute delta for all, update weight for all.
    	//Loss for each instance and print out average
    	for (int epoch = 0; epoch < maxEpoch; ++epoch) {
    	 Collections.shuffle(trainingSet,random);
    	 double loss = 0.0;
    	 //For each instance
    	 for (int i = 0; i < trainingSet.size(); ++i) {
    		 predict(trainingSet.get(i));
    		 //Compute Delta for output layer
    		 for(int j = 0; j < outputNodes.size(); ++j) {
    			 outputNodes.get(j).calculateDelta(trainingSet.get(i).classValues.get(j),0);
    		 }
    		 //Compute delta for hidden layer
    		 for (int j = 0; j < hiddenNodes.size(); ++j) {
    			 double DeltaHelper = 0.0;
    			 //compute the sum part in delta
    			 for (int k = 0; k < outputNodes.size(); ++k) {
    				 DeltaHelper += outputNodes.get(k).computingDeltaHelper(j);
    			 }
    			 hiddenNodes.get(j).calculateDelta(0, DeltaHelper);
    		 }
    		 //Update weights for output layer
    		 for(int j = 0; j < outputNodes.size(); ++j) {
    			 outputNodes.get(j).updateWeight(learningRate);
    		 }
    		 //Update weights for hidden layer
    		 for (int j = 0; j < hiddenNodes.size(); ++j) {
    			 hiddenNodes.get(j).updateWeight(learningRate);
    		 }
    	 }
		 //Forward pass again and compute the loss
		 for(int i = 0; i < trainingSet.size(); ++i) {
			 loss += loss(trainingSet.get(i));
		 }

    	 double avgLoss = loss/trainingSet.size();
    	 System.out.print("Epoch: "+epoch+", Loss: "+String.format("%.3e", avgLoss)+"\n");
    }
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // TODO: add code here, cross-entropy
    	predict(instance);
    	double loss = 0.0;
    	for (int i = 0; i < outputNodes.size(); i++) {
    		loss += instance.classValues.get(i) * Math.log(outputNodes.get(i).getOutput());
    	}
    	loss = -loss;
        return loss;
    }
}
