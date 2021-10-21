#include"neuralNet.h"
#include <stdio.h>
#include <inttypes.h>
#include <time.h>

int main(void){
	printf("start\n");
	Model m;
	printf("size of Model : %.2fkb\n", sizeof(m)/1000.0);
	modelSetup(&m, 8, 2, 8, 16, 128, 256, 128, 32, 4);
	modelSetFunctions(&m, XAVIER_UNIFORM, RELU, 0, SIGMOID, 1, HUBER);
	float input[2] = {12.7, 3.2};
	float expected[4] = {2.2, 1.6, 1.3, 4.7};
	clock_t timer = clock();
	uint32_t i;
	for (i = 0;i<8;++i){
		clock_t timeCurrent = clock();
		modelPass(&m, input, expected);
		float currentTime = (double)clock()-timeCurrent;
		float totalTime = (double)clock()-timer;
		printf("completed pass %u in %fms Total Elapsed Time: %fms\n",i,(currentTime/CLOCKS_PER_SEC)*1000,(totalTime/CLOCKS_PER_SEC)*1000);
	}
	float* output = modelOutput(&m);
	//printState(&m);
	modelClose(&m);
	printf("exit\n");
	return 0;
}
