# Back-Propagation-for-Handwritten-Digit-Recognition

This project builds a 2-layer, feed-forward neural networkand trains it 
using the back-propagation algorithm. The problem that the neural network will
handle is a multi-class classification problem for recognizing images of handwritten digits. All
inputs to the neural network will be numeric. The neural network has one hidden layer. The
network is fully connected between consecutive layers, meaning each unit, which we’ll call a node,
in the input layer is connected to all nodes in the hidden layer, and each node in the hidden layer
is connected to all nodes in the output layer. Each node in the hidden layer and the output layer
will also have an extra input from a “bias node" that has constant value +1. So, we can consider
both the input layer and the hidden layer as containing one additional node called a bias node.
All nodes in the hidden layer (except for the bias node) should use the ReLU activation function,
while all the nodes in the output layer should use the Softmax activation function. The initial
weights of the network will be set randomly (already implemented in the skeleton code). Assuming
that input examples (called instances in the code) have m attributes (hence there are m input
nodes, not counting the bias node) and we want h nodes (not counting the bias node) in the
hidden layer, and o nodes in the output layer, then the total number of weights in the network is
(m +1)h between the input and hidden layers, and (h+1)o connecting the hidden and output
layers. The number of nodes to be used in the hidden layer will be given as input.

Input: java DigitClassifier \<numHidden> \<learnRate> \<maxEpoch>
\<trainFile> \<testFile>

E.g.
java DigitClassifier 5 0.01 100 train1.txt test1.txt
