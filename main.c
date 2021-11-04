#include"neuralNet.h"
#include <stdio.h>
#include <inttypes.h>
#include <time.h>

int main(void){
	printf("start\n");
	Model m;
	printf("size of Model : %.2fkb\n", sizeof(m)/1000.0);
	modelSetup(&m, 4, 1, 16, 16, 1);
	modelSetFunctions(&m, XAVIER_UNIFORM, RELU, 0, SIGMOID, 1, HUBER);
	printf("Model Setup completed\n");
	modelTrain(&m, readDataSet("squares.txt"));
	modelClose(&m);
	printf("exit\n");
	return 0;
}
