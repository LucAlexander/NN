#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <inttypes.h>

#define MAX_LAYERS 8
#define MAX_NODES 32
#define E 2.71828
#define RELU_LEAK_A 0.01
#define HUBER_DELTA 1

typedef struct Model{
	uint32_t numLayers;
	uint32_t outputIndex;
	uint32_t numNodes[MAX_LAYERS];
	float node[MAX_LAYERS][MAX_NODES];
	float bias[MAX_LAYERS][MAX_NODES];
	/* WEIGHT DIMENSIONS
	 * layers x nodes in layer x input weights
	 * first index in layers is first hidden layer not input
	 */
	float weight[MAX_LAYERS][MAX_NODES][MAX_NODES];
	float expectedOutput[MAX_NODES];
	// WEIGHT GEN enum
	uint32_t weightInit;
	// ACTIVATION enum
	uint32_t hiddenActivation;
	uint32_t outputActivation;
	// LOSS enum
	uint32_t lossFunction;
	float loss;
	/* FLAG BITS
	 * hidden activation post operation
	 * output activation post operation
	 */
	uint8_t flags;
}Model;

enum WEIGHT_INIT{
	HE,
	HE_UNIFORM,
	XAVIER,
	XAVIER_UNIFORM,
	XAVIER_NORMAL,
	UNIFORM
}WEIGHT_INIT;

enum ACTIVATION{
	SIGMOID,
	SOFTMAX,
	BINARY_STEP,
	RELU,
	RELU_LEAKY,
	TANH
}ACTIVATION;

enum LOSS{
	MAE,
	MSE,
	MBE,
	MSLE,
	HUBER,
	BINARY_CROSS_ENTROPY,
	HINGE
}LOSS;

void modelSetup(Model* m, uint32_t n, ...);
void modelInitialConditions(Model* m);
float modelCallWeightInitFunction(uint32_t function, uint32_t n, uint32_t n0);
void modelSetFunctions(Model* m, uint32_t weightInit, uint32_t hidden, uint8_t postHidden, uint32_t output, uint8_t postOutput, uint32_t lossFunction);

void modelPass(Model* m, float input[], float expectedOutput[]);
void modelNodesPass(Model* m, uint32_t layer, uint32_t nodeCount, uint32_t prevNodeCount);
float modelCalculateNode(Model* m, uint32_t layer, uint32_t node, uint32_t n0);
float modelActivationFunction(Model* m,float nodeVal, uint32_t layerIndex, uint32_t nodeCount);
void modelActivationFunctionPost(Model* m, uint32_t layerIndex, uint32_t nodeCount);
float modelCallActivationFunction(Model* m, uint32_t function, float nodeVal, uint32_t i, uint32_t n);
float modelCallLossFunction(uint32_t function, uint32_t n, float output[], float expected[]);

float* modelOutput(Model* m);
void printState(Model* m);

float weightInitHe(uint32_t sizeL, uint32_t sizeLPrev);
float weightInitHeUniform(uint32_t sizeL, uint32_t sizeLPrev);
float weightInitXavier(uint32_t sizeL, uint32_t sizeLPrev);
float weightInitXavierUniform(uint32_t sizeL, uint32_t sizeLPrev);
float weightInitXavierNormal(uint32_t sizeL, uint32_t sizeLPrev);
float weightInitUniform(uint32_t sizeL, uint32_t sizeLPrev);

void activationSoftmax(Model* m, uint32_t layer, uint32_t n);
float activationSigmoid(float x);
float activationBinaryStep(float x);
float activationReLuLeaky(float x);
float activationReLu(float x);
float activationTanH(float x);

float dmax(float a, float b);
float dmin(float a, float b);

// REGRESSION LOSS
float lossMeanAbsoluteError(uint32_t n, float* output, float* expected);
float lossMeanSquaredError(uint32_t n, float* output, float* expected);
float lossMeanBiasError(uint32_t n, float* output, float* expected);
float lossMeanSquaredLogError(uint32_t n, float* output, float* expected);
float lossHuber(uint32_t n, float* output, float* expected);

// CLASSIFICATION LOSS
float lossBinaryCrossEntropy(uint32_t n, float* output, float* expected);
float lossHinge(uint32_t n, float* output, float* expected);

#endif
