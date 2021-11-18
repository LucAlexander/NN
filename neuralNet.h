#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <inttypes.h>
#include <pthread.h>

#define MAX_LAYERS 8
#define MAX_NODES 1024
#define E 2.71828
#define RELU_LEAK_A 0.01
#define HUBER_DELTA 1.35
#define LEARNING_RATE 0.0001
#define READ_BUFFER_SIZE 1024
#define BATCH_SIZE 1

typedef struct Model{
	uint32_t numLayers;
	uint32_t outputIndex;
	uint32_t numNodes[MAX_LAYERS];
	float node[MAX_LAYERS][MAX_NODES];
	float bias[MAX_LAYERS][MAX_NODES];
	float delta[MAX_LAYERS][MAX_NODES];
	float losses[MAX_NODES];
	/* WEIGHT DIMENSIONS
	 * layers x nodes in layer x input weights
	 * first index in layers is first hidden layer not input
	 * [i][k][t] = layers*(k+(t*numNodes))+i
	 */
	float* weight;
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

typedef struct NodeThreadParams{
	Model* m;
	uint32_t i;
	uint32_t k;
	uint32_t n;
	uint32_t n0;
	pthread_mutex_t* lock;
}NodeThreadParams;

typedef struct DataSet{
	float** X;
	float** Y;
	uint32_t n;
	uint32_t m;
}DataSet;

DataSet* readDataSet(const int8_t* fileName);
uint32_t getDataVectorLength(int8_t buffer[]);
float* parseFloatVector(int8_t buffer[], uint32_t n);
void closeDataSet(DataSet* d);

void setNodeThreadParams(NodeThreadParams* arg, Model* m, uint32_t i, uint32_t k, uint32_t n, uint32_t n0, pthread_mutex_t* lock);

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
void modelClose(Model* m);

void modelTrain(Model* m, DataSet* d);
void modelPass(Model* m, float input[], float expectedOutput[]);
void modelNodesPass(Model* m, uint32_t layer, uint32_t nodeCount, uint32_t prevNodeCount);
float modelCalculateNode(Model* m, uint32_t layer, uint32_t node, uint32_t n, uint32_t n0);
float modelActivationFunction(Model* m,float nodeVal, uint32_t layerIndex, uint32_t nodeCount);
void modelActivationFunctionPost(Model* m, uint32_t layerIndex, uint32_t nodeCount);
float modelCallActivationFunction(Model* m, uint32_t function, float nodeVal, uint32_t i, uint32_t n);
float modelCallLossFunction(uint32_t function, float* losses, uint32_t n, float output[], float expected[]);

void* modelNodeThread(void* arg);

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

float maxf(float a, float b);
float minf(float a, float b);
float absf(float a);

int64_t max(int64_t a, int64_t b);
int64_t min(int64_t a, int64_t b);

// REGRESSION LOSS
float lossMeanAbsoluteError(uint32_t n, float* losses, float* output, float* expected);
float lossMeanSquaredError(uint32_t n, float* losses, float* output, float* expected);
float lossMeanBiasError(uint32_t n, float* losses, float* output, float* expected);
float lossMeanSquaredLogError(uint32_t n, float* losses, float* output, float* expected);
float lossHuber(uint32_t n, float* losses, float* output, float* expected);

// CLASSIFICATION LOSS
float lossBinaryCrossEntropy(uint32_t n, float* losses, float* output, float* expected);
float lossHinge(uint32_t n, float* losses, float* output, float* expected);

// LOSS PARTIAL DERIVATIVES
float dldxMeanAbsoluteError(uint32_t n, float x, float y);
float dldxMeanSquaredError(uint32_t n, float x, float y);
float dldxMeanBiasError(uint32_t n, float x, float y);
float dldxMeanSquaredLogError(uint32_t n, float x, float y);
float dldxHuber(uint32_t n, float x, float y);
float dldxBinaryCrossEntropy(uint32_t n, float x, float y);
float dldxHinge(uint32_t n, float x, float y);

// ACTIVATION PARTIAL DERIVATIVES
// NODE SOFTMAX AND BINARY STEP MISSING
float ddxSigmoid(float x);
float ddxReLu(float x);
float ddxReLuLeaky(float x);
float ddxTanH(float x);

// ERROR PROPOGATION
float callLossFunctionDerivative(uint32_t function, uint32_t n, float x, float y);
float callActivationFunctionDerivative(uint32_t function, float x);
void modelBackpropogation(Model* m, float* expected);
void modelGenerateHiddenDeltas(Model* m, uint32_t i, uint32_t n, uint32_t ni);
void modelUpdateWeights(Model* m, float* input);

#endif
