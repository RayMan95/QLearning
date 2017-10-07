/*
 * CNeuralNet.h
 *
 *  Created on: 26 Dec 2013
 *      Author: benjamin
 */

#ifndef CNEURALNET_H_
#define CNEURALNET_H_
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include <stdint.h>


typedef unsigned int uint;

class CNeuralNet {
private: 
	std::vector<double> _inputs, _hidden, _outputs;
	std::vector<double> _input_weights, _output_weights;
	const double _lRate, _MSECutoff;
protected:
	void feedForward(const std::vector<double> inputs); //you may modify this to do std::vector<double> if you want
	void propagateErrorBackward(const std::vector<double> desiredOutput); //you may modify this to do std::vector<double> if you want
	double meanSquaredError(const std::vector<double> desiredOutput); //you may modify this to do std::vector<double> if you want
public:
	CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff);
	void initWeights();
	void train(const std::vector<std::vector<double>> inputs, const std::vector<std::vector<double>> outputs, uint trainingSetSize); //you may modify this to do std::vector<std::vector<double> > or do boost multiarray or something else if you want
	uint classify(const std::vector<double> input); //you may modify this to do std::vector<double> if you want
	double getOutput(uint index) const;
	virtual ~CNeuralNet();
};

#endif /* CNEURALNET_H_ */
