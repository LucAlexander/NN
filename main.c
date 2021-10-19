#include"neuralNet.h"
#include <stdio.h>
#include <inttypes.h>
#include <time.h>

int main(void){
	printf("start\n");
	Model m;
	printf("size of Model : %.2fkb\n", sizeof(m)/1000.0);
	printf("sizeof weight : %.2fkb\n",sizeof(m.weight)/1000.0);
	modelSetup(&m, 4, 2, 3, 3, 2);
	modelSetFunctions(&m, XAVIER_UNIFORM, TANH, 0, SOFTMAX, 1, HUBER);
	float input[2] = {5.6, 3.2};
	float expected[2] = {2.0, 1.0};
	clock_t timer = clock();
	modelPass(&m, input, expected);
	timer = clock()-timer;
	float* output = modelOutput(&m);
	printf("completed pass in %f ms\nnet state:\n",timer);
	printState(&m);
	printf("exit\n");
	return 0;
}
