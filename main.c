#include"neuralNet.h"
#include <stdio.h>
#include <inttypes.h>
#include <time.h>

int main(void){
	printf("start\n");
	Model m;
	printf("size of Model : %.2fkb\n", sizeof(m)/1000.0);
	printf("sizeof weight : %.2fkb\n",sizeof(m.weight)/1000.0);
	modelSetup(&m, 8, 2, 4, 4, 4, 4, 4, 4, 4);
	modelSetFunctions(&m, XAVIER_UNIFORM, RELU, 0, SIGMOID, 1, HUBER);
	float input[2] = {12.7, 3.2};
	float expected[4] = {2.2, 1.6, 1.3, 4.7};
	clock_t timer = clock();
	modelPass(&m, input, expected);
	float timeToComplete = (double)clock()-timer;
	float* output = modelOutput(&m);
	printf("completed pass in %f ms\nnet state:\n",timeToComplete/CLOCKS_PER_SEC);
	printState(&m);
	printf("exit\n");
	return 0;
}
