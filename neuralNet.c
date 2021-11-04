#include "neuralNet.h"
#include "rng.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>

void modelSetup(Model* m, uint32_t n, ...){
	startRandSeed();
	n = min(n, MAX_LAYERS);
	m->numLayers = n;
	m->outputIndex = n-1;
	memset(m->numNodes, 0, sizeof(m->numNodes));
	memset(m->node, 0, sizeof(m->node));
	memset(m->bias, 0, sizeof(m->bias));
	m->weightInit = HE_UNIFORM;
	m->hiddenActivation = RELU;
	m->outputActivation = SIGMOID;
	m->lossFunction = HUBER;
	m->loss = 0.0;
	m->flags = 0;
	va_list v;
	va_start(v, n);
	uint32_t maxCount = 0;
	uint32_t i = 0;
	while(i<n){
		m->numNodes[i] = va_arg(v, uint32_t);
		maxCount = max(maxCount, m->numNodes[i]);
		i++;
	}
	va_end(v);
	maxCount = min(maxCount, MAX_NODES);
	m->weight = calloc((n*maxCount*maxCount),sizeof(float));
}

void modelInitialConditions(Model* m){
	uint32_t i, k, t, n;
	uint32_t n0 = m->numNodes[0];
	for (i = 1;i<m->numLayers;++i){
		n = m->numNodes[i];
		for (k = 0;k<n;++k){
			for (t = 0;t<n0;++t){
				m->weight[m->numLayers*(k+(t*n))+i-1] = modelCallWeightInitFunction(m->weightInit, n, n0);
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

void modelClose(Model* m){
	free(m->weight);
	m->weight = NULL;
}

void modelTrain(Model* m, DataSet* d){
	uint32_t i;
	clock_t timer = clock();
	for (i=0;i<d->n;++i){
		clock_t timeCurrent = clock();
		modelPass(m, d->X[i], d->Y[i]);
		float currentTime = (double)clock()-timeCurrent;
		float totalTime = (double)clock()-timer;
		printf("completed pass %u in %.0fms Total Elapsed Time: %.0fms | Loss: %.2f\n",i,(currentTime/CLOCKS_PER_SEC)*1000,(totalTime/CLOCKS_PER_SEC)*1000,m->loss);
	}
	closeDataSet(d);
}

void modelPass(Model* m, float input[], float expectedOutput[]){
	uint32_t i, n, n0;
	n0 = m->numNodes[0];
	memcpy(m->node[0], input, sizeof(float)*n0);
	for (i = 1;i<m->numLayers;++i){
		n = m->numNodes[i];
		modelNodesPass(m, i, n, n0);
		n0 = n;
	}
	float output[n];
	memcpy(output, m->node[n], sizeof(float)*n);
	m->loss = modelCallLossFunction(m->lossFunction, n, output, expectedOutput);
	//gradientDescent(m, input, expectedOutput, output);
}

void modelNodesPass(Model* m, uint32_t i, uint32_t n, uint32_t n0){
	pthread_mutex_t lock;
	pthread_mutex_init(&lock, NULL);
	pthread_t threads[n];
	uint32_t argSize = sizeof(NodeThreadParams);
	NodeThreadParams* args[n];
	uint32_t k, err;
	for (k = 0;k<n;++k){
		NodeThreadParams* arg = malloc(argSize);
		memset(arg, 0, argSize);
		args[k] = arg;
		setNodeThreadParams(arg, m, i, k, n, n0, &lock);
		err = pthread_create(&threads[k], NULL, &modelNodeThread, (void*)arg);
		if (err){
			printf("Could not create thread %u, exit code %u\n",k,err);
		}
	}
	for (k = 0;k<n;++k){
		pthread_join(threads[k], NULL);
	}
	pthread_mutex_destroy(&lock);
	for (k = 0;k<n;++k){
		free(args[k]);
		args[k] = NULL;
	}
	modelActivationFunctionPost(m, i, n);
}

void setNodeThreadParams(NodeThreadParams* arg, Model* m, uint32_t i, uint32_t k, uint32_t n, uint32_t n0, pthread_mutex_t* lock){
	arg->m = m;
	arg->i = i;
	arg->k = k;
	arg->n = n;
	arg->n0 = n0;
	arg->lock = lock;
}

void* modelNodeThread(void* args){
	uint32_t i, k, n, n0;
	float nodeVal;
	NodeThreadParams* argList = args;
	Model* m = argList->m;
	i = argList->i;
	k = argList->k;
	n = argList->n;
	n0 = argList->n0;
	pthread_mutex_t* lock = argList->lock;
	pthread_mutex_lock(lock);
	nodeVal = modelCalculateNode(m, i, k, n, n0);
	nodeVal += m->bias[i][k];
	m->node[i][k] = modelActivationFunction(m, nodeVal, i, n);
	pthread_mutex_unlock(lock);
	pthread_exit(NULL);
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

float modelCalculateNode(Model* m, uint32_t i, uint32_t k, uint32_t n, uint32_t n0){
	float nodeVal = 0.0;
	uint32_t t;
	for (t = 0;t<n0;++t){
		//nodeVal += m->weight[i-1][k][t]*m->node[i-1][t];
		nodeVal += m->weight[m->numLayers*(k+(t*n))+i-1]*m->node[i-1][t];
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
		ymax = maxf(ymax, yi);
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
	return maxf(0, x);
}

float activationTanH(float x){
	return 2*activationSigmoid(2*x)-1;
}

float maxf(float a, float b){
	if (a>b){
		return a;
	}
	return b;
}

float minf(float a, float b){
	if (a<b){
		return a;
	}
	return b;
}

int64_t max(int64_t a, int64_t b){
	if (a<b){
		return b;
	}
	return a;
}

int64_t min(int64_t a, int64_t b){
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
		sum+=maxf(0, expected[i]-output[i]+1);
	}
	return sum/n;
}

float dldxMeanAbsoluteError(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float sum = 0.0;
	for (i=0;i<n;++i){
		float a = output[i]-expected[i];
		sum += -var*(a/abs(a));
	}
	return sum/n;
}

float dldxMeanSquaredError(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float sum = 0.0;
	for (i=0;i<n;++i){
		sum += -var*(expected[i]-output[i]);
	}
	return (2*sum)/n;
}

float dldxMeanBiasError(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float sum = 0.0;
	for (i=0;i<n;++i){
		sum += -var;
	}
	return sum/n;
}

float dldxMeanSquaredLogError(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float sum = 0.0;
	for (i=0;i<n;++i){
		float a = (-2*var)/output[i];
		a = (log(expected[i])-log(output[i]));
		sum += a;
	}
	return sum/n;
}

float dldxHuber(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		float a = expected[i]-output[i];
		if (abs(a) <= HUBER_DELTA){
			sum += -var*a;
			continue;
		}
		sum += (-var*HUBER_DELTA*a)/abs(a);
	}
	return sum/n;
}

float dldxBinaryCrossEntropy(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float y, x;
	float sum = 0.0;
	for (i = 0;i<n;++i){
		y = expected[i];
		x = output[i];
		sum += (y*var)/x;
		sum += ((1-y)*var)/(1-x);
	}
	return -(sum/n);
}

float dldxHinge(uint32_t n, float* output, float* expected, float var){
	uint32_t i;
	float sum = 0.0;
	for (i=0;i<n;++i){
		sum += maxf(0, -var);
	}
	return sum/n;
}

void gradientDescent(Model* m, float* input, float* expected, float* output){
	uint32_t n = m->numNodes[m->outputIndex];
	float dx = dldw(input, expected, output, n);
	updateWeights(m, dx);
	dx = dldb(expected, output, n);
	updateBiases(m, dx);
	//TODO fix this, it does the opposite of optimize
}

void updateWeights(Model* m, float dx){
	uint32_t i, k, t, n, index;
	uint32_t n0 = m->numNodes[0];
	for (i = 1;i<m->numLayers;++i){
		n = m->numNodes[i];
		for (k = 0;k<n;++k){
			for (t = 0;t<n0;++t){
				index = m->numLayers*(k+(t*n)+i-1);
				m->weight[index] -= LEARNING_RATE*dx;
			}
		}
		n0=n;
	}
}

void updateBiases(Model* m, float dx){
	uint32_t i, k, n;
	for (i = 1;i<m->numLayers;++i){
		n = m->numNodes[i];
		for (k = 0;k<n;++k){
			m->bias[i][k] -= LEARNING_RATE*dx;
		}
	}
}

float dldw(float* input, float* expected, float* output, uint32_t n){
	float s = 0;
	uint32_t i;
	for (i = 0;i<n;++i){
		s+=input[i]+(expected[i]-output[i]);
	}
	return (2/n)*s;
}
	
float dldb(float* expected, float* output, uint32_t n){
	float s = 0;
	uint32_t i;
	for (i = 0;i<n;++i){
		s += (expected[i]-output[i]);
	}
	return (2/n)*s;
}

DataSet* readDataSet(const int8_t* fileName){
	FILE* fin;
	fin = fopen(fileName, "r");
	if (fin==NULL){
		printf("Data set file %s not found\n",fileName);
		fclose(fin);
		return NULL;
	}
	DataSet* d = malloc(sizeof(DataSet));
	d->X=NULL;
	d->Y=NULL;
	d->n=0;
	d->m=0;
	int8_t buffer[READ_BUFFER_SIZE];
	while(fgets(buffer, READ_BUFFER_SIZE, fin)!=NULL){
		if (d->m==0){
			d->m=getDataVectorLength(buffer);
		}
		d->n++;
	}
	fclose(fin);
	d->n/=2;
	d->X=malloc(d->n*sizeof(float*));
	d->Y=malloc(d->n*sizeof(float*));
	uint32_t i = 0;
	for (i = 0;i<d->n;++i){
		d->X[i]=malloc(d->m*sizeof(float));
		d->Y[i]=malloc(d->m*sizeof(float));
	}
	fin = fopen(fileName, "r");
	i=0;
	while(fgets(buffer, READ_BUFFER_SIZE, fin)!=NULL){
		d->X[i]=parseFloatVector(buffer, d->m);
		fgets(buffer, READ_BUFFER_SIZE, fin);
		d->Y[i++]=parseFloatVector(buffer, d->m);
	}
	fclose(fin);
	return d;
}

uint32_t getDataVectorLength(int8_t buffer[]){
	uint32_t i=0;
	uint32_t n=0;
	uint8_t f = 0;
	int8_t c = ' ';
	while (c!='\0'&&c!='\n'){
		c = buffer[i++];
		if(c==','){
			n++;
		}
		f = 1;
	}
	return n+f;
}

float* parseFloatVector(int8_t buffer[], uint32_t n){
	uint32_t i=0;
	uint32_t k=0;
	uint32_t j=0;
	int8_t c = ' ';
	int8_t s[sizeof(float)*8];
	// Logic behind previous line is that 
	// you're not gonna have a float
	// represented in the input buffer
	// (base ten) wider than the bit width
	// of a float
	float* vector = malloc(sizeof(float)*n);
	while(c!='\n'&&c!='\0'){
		c=buffer[i++];
		if (c==','){
			s[k] = '\0';
			vector[j++]=atof(s);
			memset(buffer, 0, strlen(buffer));
			k = 0;
			continue;
		}
		if (c!=' '){
			s[k++] = c;
		}
	}
	s[k]='\0';
	vector[j]=atof(s);
	return vector;
}

void closeDataSet(DataSet* d){
	if (d==NULL){
		return;
	}
	uint32_t i;
	for (i=0;i<d->n;++i){
		free(d->X[i]);
		free(d->Y[i]);
		d->X[i] = NULL;
		d->Y[i] = NULL;
	}
	free(d->X);
	free(d->Y);
	d->X = NULL;
	d->Y = NULL;
	free(d);
	d=NULL;
}
