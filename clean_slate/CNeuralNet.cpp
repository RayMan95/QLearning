/*
                                                                           
   (               )                                        )              
 ( )\     )     ( /(       (                  (  (     ) ( /((             
 )((_) ( /(  (  )\())`  )  )(   (  `  )   (   )\))( ( /( )\())\  (   (     
((_)_  )(_)) )\((_)\ /(/( (()\  )\ /(/(   )\ ((_))\ )(_)|_))((_) )\  )\ )  
 | _ )((_)_ ((_) |(_|(_)_\ ((_)((_|(_)_\ ((_) (()(_|(_)_| |_ (_)((_)_(_/(  
 | _ \/ _` / _|| / /| '_ \) '_/ _ \ '_ \/ _` |/ _` |/ _` |  _|| / _ \ ' \)) 
 |___/\__,_\__||_\_\| .__/|_| \___/ .__/\__,_|\__, |\__,_|\__||_\___/_||_|  
                    |_|           |_|         |___/                         

 For more information on back-propagation refer to:
 Chapter 18 of Russel and Norvig (2010).
 Artificial Intelligence - A Modern Approach.
 */

#include "CNeuralNet.h"
#include "utils.h"

using namespace std;

/**
 The constructor of the neural network. This constructor will allocate memory
 for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 and output layers.
*/
CNeuralNet::CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff) :
	_lRate(lRate), _MSECutoff(mse_cutoff)
{
	_inputs = vector<double>(inputLayerSize);
	_hidden = vector<double>(hiddenLayerSize);
	_outputs = vector<double>(outputLayerSize);
}
/**
 The destructor of the class. All allocated memory will be released here
*/
CNeuralNet::~CNeuralNet() {
	//TODO
}
/**
 Method to initialize the both layers of weights to random numbers
*/
void CNeuralNet::initWeights(){
	int in_size = _inputs.size(), out_size = _outputs.size();

	for (int i = 0; i < in_size; ++i)
	{
		_input_weights[i] = RandFloat();
	}
	for (int i = 0; i < in_size; ++i)
	{
		_output_weights[i] = RandFloat();
	}
}
/**
 This is the forward feeding part of back propagation.
 1. This should take the input and copy the memory (use memcpy / std::copy)
 to the allocated _input array.
 2. Compute the output of at the hidden layer nodes 
 (each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
 3. Repeat step 2, but this time compute the output at the output layer
*/
void CNeuralNet::feedForward(const std::vector<double> inputs) {

	std::copy(inputs.begin(), inputs.end(), _inputs.begin());
	
	// input -> hidden nodes
	int i = 0, hidden_size = _hidden.size();
	while (i < hidden_size) {
		double  sum = 0;
		for (int j = 0; j < _inputs.size(); ++j) {
			sum += _input_weights[i] * inputs[i];
		}
		// 'fast sigmoid' : f(x) = x / (1 + abs(x))
		_hidden[i] = sum / (1 + abs(sum));
	}

	// hidden -> output nodes
	int output_size = _outputs.size();
	i = 0;
	while (i < output_size) {
		double  sum = 0;
		for (int j = 0; j < _hidden.size(); ++j) {
			sum += _output_weights[i] * _hidden[i];
		}

		_outputs[i] = sum / (1 + abs(sum));
	}
}
/**
 This is the actual back propagation part of the back propagation algorithm
 It should be executed after feeding forward. Given a vector of desired outputs
 we compute the error at the hidden and output layers (allocate some memory for this) and
 assign 'blame' for any error to all the nodes that fed into the current node, based on the
 weight of the connection.
 Steps:
 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs)
    for each output
 2. Compute the error at the hidden layer: sigmoid_d(hidden) * 
	sum(weights_o_h * difference between expected output and computed output at output layer)
	for each hidden layer node
 3. Adjust the weights from the hidden to the output layer: learning rate * error at the output layer * error at the hidden layer
    for each connection between the hidden and output layers
 4. Adjust the weights from the input to the hidden layer: learning rate * error at the hidden layer * input layer node value
    for each connection between the input and hidden layers
 5. REMEMBER TO FREE ANY ALLOCATED MEMORY WHEN YOU'RE DONE (or use std::vector ;)
*/
void CNeuralNet::propagateErrorBackward(const std::vector<double> desiredOutput){
	//TODO
}
/**
This computes the mean squared error
A very handy formula to test numeric output with. You may want to commit this one to memory
*/
double CNeuralNet::meanSquaredError(const std::vector<double> desiredOutput){
	int sum = 0, outputLayerSize = _outputs.size(), err = 0;
	
	for (int i = 0; i < outputLayerSize; ++i) {
		err = desiredOutput[i] - _outputs[i];
		sum += err*err;
	}
	return sum / outputLayerSize;
}
/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
	for each training pattern:
	  feed forward
	  propagate backward
	until the MSE becomes suitably small
*/
void CNeuralNet::train(const std::vector<std::vector<double>> inputs,
	const std::vector<std::vector<double>> outputs, uint trainingSetSize) {
	//TODO

}
/**
Once our network is trained we can simply feed it some input through the feed forward
method and take the maximum value as the classification
*/
uint CNeuralNet::classify(const std::vector<double> input){
	return 1; //TODO: fix me
}
/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(uint index) const{
	return _outputs[index];
}