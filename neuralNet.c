#include "neuralNet.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>

void modelSetup(Model* m, uint32_t n, ...){
	startRandSeed();
	m->numLayers = n;
	m->outputIndex = n-1;
	memset(m->numNodes, 0, sizeof(m->numNodes));
	memset(m->node, 0, sizeof(m->node));
	memset(m->bias, 0, sizeof(m->bias));
	memset(m->weight, 0, sizeof(m->weight));
	m->weightInit = HE_UNIFORM;
	m->hiddenActivation = RELU;
	m->outputActivation = SIGMOID;
	m->lossFunction = HUBER;
	m->loss = 0.0;
	m->flags = 0;
	va_list v;
	va_start(v, n);
	while(n>0){
		m->numNodes[n-1] = va_arg(v, uint32_t);
		n--;
	}
	va_end(v);
}

void modelInitialConditions(Model* m){
	uint32_t i, k, t, n;
	uint32_t n0 = m->numNodes[0];
	for (i = 1;i<m->numLayers;++i){
		n = m->numNodes[i];
		for (k = 0;k<n;++k){
			for (t = 0;t<n0;++t){
				m->weight[i-1][k][t] = modelCallWeightInitFunction(m->weightInit, n, n0);
			}
		}
		n0 = n;
	}
}

float modelCallWeightInitFunction(uint32_t function, uint32_t n, uint32_t n0){
	switch(function){
		case HE:
			return weightInitHe(n, n0);
		break;
		case HE_UNIFORM:
			return weightInitHeUniform(n, n0);
		break;
		case XAVIER:
			return weightInitXavier(n, n0);
		break;
		case XAVIER_UNIFORM:
			return weightInitXavierUniform(n, n0);
		break;
		case XAVIER_NORMAL:
			return weightInitXavierNormal(n, n0);
		break;
		case UNIFORM:
			return weightInitUniform(n, n0);
		break;
	}
	return 1;
}

void modelSetFunctions(Model* m, uint32_t weightInit, uint32_t hidden, uint8_t postHidden, uint32_t output, uint8_t postOutput, uint32_t lossFunction){
	m->weightInit = weightInit;
	m->hiddenActivation = hidden;
	m->outputActivation = output;
	m->flags |= postHidden | (postOutput<<1);
	m->lossFunction = lossFunction;
	modelInitialConditions(m);
}

void modelPass(Model* m, float input[], float expectedOutput[]){
	uint32_t i, n, n0;
	n0 = m->numNodes[0];
	memcpy(m->node[0], input, sizeof(float)*n0);
	memcpy(m->expectedOutput, expectedOutput, sizeof(float)*m->numNodes[m->outputIndex]);
	for (i = 1;i<m->numLayers;++i){
		n = m->numNodes[i];
		modelNodesPass(m, i, n, n0);
		n0 = n;
	}
	float output[n];
	memcpy(output, m->node[n], sizeof(float)*n);
	m->loss = modelCallLossFunction(m->lossFunction, n, output, expectedOutput);
	// TODO OPTIMIZATION
}

void modelNodesPass(Model* m, uint32_t i, uint32_t n, uint32_t n0){
	float nodeVal;
	uint32_t k;
	for (k = 0;k<n;++k){
		nodeVal = modelCalculateNode(m, i, k, n0);
		nodeVal += m->bias[i][k];
		m->node[i][k] = modelActivationFunction(m, nodeVal, i, n);
	}
	modelActivationFunctionPost(m, i, n);
}

float modelActivationFunction(Model* m, float nodeVal, uint32_t i, uint32_t n){
	if (i!=m->outputIndex&&m->flags&1==0){
		return modelCallActivationFunction(m, m->hiddenActivation, nodeVal, i, n);
	}
	if (i==m->outputIndex&&m->flags&2==0){
		return modelCallActivationFunction(m, m->outputActivation, nodeVal, i, n);
	}
	return nodeVal;
}

void modelActivationFunctionPost(Model* m, uint32_t i, uint32_t n){
	if (i!=m->outputIndex&&m->flags&1!=0){
		modelCallActivationFunction(m, m->hiddenActivation, 0, i, n);
	}
	if (i==m->outputIndex&&m->flags&2!=0){
		modelCallActivationFunction(m, m->hiddenActivation, 0, i, n);
	}
}

float modelCallActivationFunction(Model* m, uint32_t function, float nodeVal, uint32_t i, uint32_t n){
	switch(function){
		case SIGMOID:
			return activationSigmoid(nodeVal);
		break;
		case SOFTMAX:
			activationSoftmax(m, i, n);
			return 0;
		break;
		case BINARY_STEP:
			return activationBinaryStep(nodeVal);
		break;
		case RELU:
			return activationReLu(nodeVal);
		break;
		case RELU_LEAKY:
			return activationReLuLeaky(nodeVal);
		break;
		case TANH:
			return activationTanH(nodeVal);
		break;
	}
	return nodeVal;
}

float modelCalculateNode(Model* m, uint32_t i, uint32_t k, uint32_t n0){
	float nodeVal = 0.0;
	uint32_t t;
	for (t = 0;t<n0;++t){
		nodeVal += m->weight[i-1][k][t]*m->node[i-1][t];
	}
	return nodeVal;
}

float modelCallLossFunction(uint32_t function, uint32_t n, float output[], float expected[]){
	switch(function){
		case MAE:
			return lossMeanAbsoluteError(n, output, expected);
		break;
		case MSE:
			return lossMeanSquaredError(n, output, expected);
		break;
		case MBE:
			return lossMeanBiasError(n, output, expected);
		break;
		case MSLE:
			return lossMeanSquaredLogError(n, output, expected);
		break;
		case HUBER:
			return lossHuber(n, output, expected);
		break;
		case BINARY_CROSS_ENTROPY:
			return lossBinaryCrossEntropy(n, output, expected);
		break;
		case HINGE:
			return lossHinge(n, output, expected);
		break;
	}
}

float* modelOutput(Model* m){
	uint32_t i, n;
	n = m->numNodes[m->outputIndex];
	float * data = malloc(sizeof(float)*n);
	for(i = 0;i<n;++i){
		data[i] = m->node[m->outputIndex][i];
	}
	return data;
}

void printState(Model* m){
	uint32_t i, k, t, n;
	for (i = 0;i<m->numLayers;++i){
		printf("LAYER %u : ", i);
		n = m->numNodes[i];
		for (k = 0;k<n;++k){
			printf("%f\t",m->node[i][k]);
		}
		printf("\n");
	}
	printf("Loss: %f\n",m->loss);
}

float weightInitUniform(uint32_t fout, uint32_t fin){
	float val = sqrt(1/fin);
	return randRangeF(-val, val);
}

float weightInitHe(uint32_t fout, uint32_t fin){
	return sqrt(2.0/fin);
}

float weightInitHeUniform(uint32_t fout, uint32_t fin){
	float val = sqrt(6.0/fin);
	return randRangeF(-val, val);
}

float weightInitXavier(uint32_t fout, uint32_t fin){
	return sqrt(1.0/fin);
}

float weightInitXavierNormal(uint32_t fout, uint32_t fin){
	return sqrt(2.0/(fout+fin));
}

float weightInitXavierUniform(uint32_t fout, uint32_t fin){
	float val = sqrt(6.0/(fout+fin));
	return randRangeF(-val, val);
}

float activationSigmoid(float x){
	return 1/(1+pow(E, -x));
}

void activationSoftmax(Model* m, uint32_t i, uint32_t n){
	uint32_t k;
	float vsum = 0.0;
	float ymax = 0.0;
	for (k = 0;k<n;++k){
		float yi = pow(E, m->node[i][k]);
		ymax = dmax(ymax, yi);
		m->node[i][k] = yi;
	}
	for (k = 0;k<n;++k){
		float vi = m->node[i][k]-ymax;
		m->node[i][k] = vi;
		vsum += vi;
	}
	for (k = 0;k<n;++k){
		m->node[i][k] = m->node[i][k]/vsum;
	}
}

float activationBinaryStep(float x){
	return (x>=0);
}

float activationReLuLeaky(float x){
	if (x>=0){
		return x;
	}
	return RELU_LEAK_A*x;
}

float activationReLu(float x){
	return dmax(0, x);
}

float activationTanH(float x){
	return 2*activationSigmoid(2*x)-1;
}

float dmax(float a, float b){
	if (a>b){
		return a;
	}
	return b;
}

float dmin(float a, float b){
	if (a<b){
		return a;
	}
	return b;
}

float lossMeanAbsoluteError(uint32_t n, float* output, float* expected){
	uint32_t i;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		sum += abs(output[i]-expected[i]);
	}
	return sum/n;
}

float lossMeanSquaredError(uint32_t n, float* output, float* expected){
	uint32_t i;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		sum += pow(expected[i]-output[i],2);
	}
	return sum/n;
}

float lossMeanBiasError(uint32_t n, float* output, float* expected){
	uint32_t i;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		sum += expected[i]-output[i];
	}
	return sum/n;
}

float lossMeanSquaredLogError(uint32_t n, float* output, float* expected){
	uint32_t i;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		sum += pow(log(expected[i])-log(output[i]), 2);
	}
	return sum/n;

}

float lossHuber(uint32_t n, float* output, float* expected){
	uint32_t i;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		float a = expected[i]-output[i];
		if (abs(a) <= HUBER_DELTA){
			sum += 0.5*pow(a, 2);
			continue;
		}
		sum += HUBER_DELTA*(abs(a)-(0.5*HUBER_DELTA));
	}
	return sum/n;
}

float lossBinaryCrossEntropy(uint32_t n, float* output, float* expected){
	uint32_t i;
	float y, x;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		y = expected[i];
		x = output[i];
		sum += y*log(x);
		sum += (1-y)*log(1-x);
	}
	return -(sum/n);
}

float lossHinge(uint32_t n, float* output, float* expected){
	uint32_t i;
	float sum = 0.0;
	for (i=0;i<n;++i){
		sum+=dmax(0, expected[i]-output[i]+1);
	}
	return sum/n;
}
