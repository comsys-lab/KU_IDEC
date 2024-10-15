#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "common/Kernel.cu"

#define INPUT_SIZE 224*224*3


/* Function to Read Alexnet Input Parameters */
void read_parameter(const char *pFileName,float *layer_parameters)
{
	FILE *fp = fopen(pFileName, "rb");
	int count = 0;
	double temp_num;
	//printf(" File FOUND : %s\n",pFileName);
	while(fscanf(fp, "%lf", &temp_num) == 1){
		layer_parameters[count] = temp_num;
		count++;
	}
	//printf("Final Count : %d\n", count);
	fclose(fp);
}

void host2gpu_alexnet(float **Alex_Layer1_Neurons,float **Alex_Layer2_Neurons,float **Alex_Layer3_Neurons,float **Alex_Layer4_Neurons,
					float **Alex_Layer5_Neurons,float **Alex_Layer6_Neurons,float **Alex_Layer7_Neurons,float **Alex_Layer8_Neurons,
                    float **Alex_Layer1_bias,float **Alex_Layer2_bias,float **Alex_Layer3_bias,float **Alex_Layer4_bias,
                    float **Alex_Layer5_bias,float **Alex_Layer6_bias,float **Alex_Layer7_bias,float **Alex_Layer8_bias,
                    float **Alex_Layer1_Weights,float **Alex_Layer2_Weights,float **Alex_Layer3_Weights,float **Alex_Layer4_Weights,
                    float **Alex_Layer5_Weights,float **Alex_Layer6_Weights,float **Alex_Layer7_Weights,float **Alex_Layer8_Weights,
                    float **Alex_Layer1_pool,float **Alex_Layer2_pool,float **Alex_Layer5_pool,
					float **Alex_Layer1_norm,float **Alex_Layer2_norm,float **Alex_Result_Neurons)
{

	float *Alex_Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));
	read_parameter("data_alexnet/input_cat1.txt", Alex_Layer1_Neurons_CPU);

	float *Alex_Layer1_bias_CPU = (float*) malloc (64 * sizeof(float));
	float *Alex_Layer2_bias_CPU = (float*) malloc (192 * sizeof(float));
	float *Alex_Layer3_bias_CPU = (float*) malloc (384 * sizeof(float));
	float *Alex_Layer4_bias_CPU = (float*) malloc (256 * sizeof(float));
	float *Alex_Layer5_bias_CPU = (float*) malloc (256 * sizeof(float));
	float *Alex_Layer6_bias_CPU = (float*) malloc (4096 * sizeof(float));
	float *Alex_Layer7_bias_CPU = (float*) malloc (4096 * sizeof(float));
	float *Alex_Layer8_bias_CPU = (float*) malloc (1000 * sizeof(float));

	float *Alex_Layer1_Weights_CPU = (float*) malloc (64*11*11*3 * sizeof(float));
	float *Alex_Layer2_Weights_CPU = (float*) malloc (192*5*5*64 * sizeof(float));
	float *Alex_Layer3_Weights_CPU = (float*) malloc (384*3*3*192 * sizeof(float));
	float *Alex_Layer4_Weights_CPU = (float*) malloc (256*3*3*384 * sizeof(float));
	float *Alex_Layer5_Weights_CPU = (float*) malloc (256*3*3*256 * sizeof(float));
	float *Alex_Layer6_Weights_CPU = (float*) malloc (4096*256*6*6 * sizeof(float));
	float *Alex_Layer7_Weights_CPU = (float*) malloc (4096*4096 * sizeof(float));
	float *Alex_Layer8_Weights_CPU = (float*) malloc (1000*4096 * sizeof(float));

	read_parameter("data_alexnet/bias1.txt", Alex_Layer1_bias_CPU);
	read_parameter("data_alexnet/bias2.txt", Alex_Layer2_bias_CPU);
	read_parameter("data_alexnet/bias3.txt", Alex_Layer3_bias_CPU);
	read_parameter("data_alexnet/bias4.txt", Alex_Layer4_bias_CPU);
	read_parameter("data_alexnet/bias5.txt", Alex_Layer5_bias_CPU);
	read_parameter("data_alexnet/bias6.txt", Alex_Layer6_bias_CPU);
	read_parameter("data_alexnet/bias7.txt", Alex_Layer7_bias_CPU);
	read_parameter("data_alexnet/bias8.txt", Alex_Layer8_bias_CPU);

	read_parameter("data_alexnet/conv1.txt", Alex_Layer1_Weights_CPU);
	read_parameter("data_alexnet/conv2.txt", Alex_Layer2_Weights_CPU);
	read_parameter("data_alexnet/conv3.txt", Alex_Layer3_Weights_CPU);
	read_parameter("data_alexnet/conv4.txt", Alex_Layer4_Weights_CPU);
	read_parameter("data_alexnet/conv5.txt", Alex_Layer5_Weights_CPU);
	read_parameter("data_alexnet/fc6.txt", Alex_Layer6_Weights_CPU);
	read_parameter("data_alexnet/fc7.txt", Alex_Layer7_Weights_CPU);
	read_parameter("data_alexnet/fc8.txt", Alex_Layer8_Weights_CPU);

    float *Alex_Layer1_Neurons_data;
	float *Alex_Layer1_bias_data, *Alex_Layer2_bias_data, *Alex_Layer3_bias_data, *Alex_Layer4_bias_data, 
			*Alex_Layer5_bias_data, *Alex_Layer6_bias_data, *Alex_Layer7_bias_data, *Alex_Layer8_bias_data;
	float *Alex_Layer1_Weights_data, *Alex_Layer2_Weights_data, *Alex_Layer3_Weights_data, *Alex_Layer4_Weights_data,
			*Alex_Layer5_Weights_data, *Alex_Layer6_Weights_data, *Alex_Layer7_Weights_data, *Alex_Layer8_Weights_data;

	cudaMalloc((void**) &Alex_Layer1_Neurons_data, INPUT_SIZE * sizeof(float)); //224*224*3
	cudaMalloc((void**) &Alex_Layer1_bias_data, 64 * sizeof(float)); //64
	cudaMalloc((void**) &Alex_Layer1_Weights_data, (64*11*11*3) * sizeof(float)); //64*11*11*3 = 23232
	cudaMalloc((void**) &Alex_Layer2_bias_data, 192 * sizeof(float)); //192
	cudaMalloc((void**) &Alex_Layer2_Weights_data, (192*5*5*64) * sizeof(float)); //192*5*5*64 = 307200
	cudaMalloc((void**) &Alex_Layer3_bias_data, 384 * sizeof(float)); //384
	cudaMalloc((void**) &Alex_Layer3_Weights_data, (384*3*3*192) * sizeof(float)); //384*3*3*192 = 663552
	cudaMalloc((void**) &Alex_Layer4_bias_data, 256 * sizeof(float)); //256
	cudaMalloc((void**) &Alex_Layer4_Weights_data, (256*3*3*384) * sizeof(float)); //256*3*3*384 = 884736
	cudaMalloc((void**) &Alex_Layer5_bias_data, 256 * sizeof(float)); //256
	cudaMalloc((void**) &Alex_Layer5_Weights_data, (256*3*3*256) * sizeof(float)); //256*3*3*256 = 442368
	cudaMalloc((void**) &Alex_Layer6_bias_data, 4096 * sizeof(float)); //4096
	cudaMalloc((void**) &Alex_Layer6_Weights_data, (4096*256*6*6) * sizeof(float)); //4096*256*6*6 = 37748736
	cudaMalloc((void**) &Alex_Layer7_bias_data, 4096 * sizeof(float)); //4096
	cudaMalloc((void**) &Alex_Layer7_Weights_data, (4096*4096) * sizeof(float)); //4096*4096 = 16777216
	cudaMalloc((void**) &Alex_Layer8_bias_data, 1000 * sizeof(float)); //1000
	cudaMalloc((void**) &Alex_Layer8_Weights_data, (1000*4096) * sizeof(float)); //1000*4096 = 4096000
	
	cudaMemcpy(Alex_Layer1_Neurons_data, Alex_Layer1_Neurons_CPU, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer1_bias_data, Alex_Layer1_bias_CPU, 64 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer1_Weights_data, Alex_Layer1_Weights_CPU, (64*11*11*3) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer2_bias_data, Alex_Layer2_bias_CPU, 192 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer2_Weights_data, Alex_Layer2_Weights_CPU, (192*5*5*64) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer3_bias_data, Alex_Layer3_bias_CPU, 384 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer3_Weights_data, Alex_Layer3_Weights_CPU, (384*3*3*192) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer4_bias_data, Alex_Layer4_bias_CPU, 256 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer4_Weights_data, Alex_Layer4_Weights_CPU, (256*3*3*384) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer5_bias_data, Alex_Layer5_bias_CPU, 256 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer5_Weights_data, Alex_Layer5_Weights_CPU, (256*3*3*256) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer6_bias_data, Alex_Layer6_bias_CPU, 4096 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer6_Weights_data, Alex_Layer6_Weights_CPU, (4096*256*6*6) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer7_bias_data, Alex_Layer7_bias_CPU, 4096 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer7_Weights_data, Alex_Layer7_Weights_CPU, (4096*4096) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer8_bias_data, Alex_Layer8_bias_CPU, 1000 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Alex_Layer8_Weights_data, Alex_Layer8_Weights_CPU, (1000*4096) * sizeof(float), cudaMemcpyHostToDevice);

	*Alex_Layer1_Neurons = Alex_Layer1_Neurons_data;

	*Alex_Layer1_bias = Alex_Layer1_bias_data;
	*Alex_Layer2_bias = Alex_Layer2_bias_data;
	*Alex_Layer3_bias = Alex_Layer3_bias_data;
	*Alex_Layer4_bias = Alex_Layer4_bias_data;
	*Alex_Layer5_bias = Alex_Layer5_bias_data;
	*Alex_Layer6_bias = Alex_Layer6_bias_data;
	*Alex_Layer7_bias = Alex_Layer7_bias_data;
	*Alex_Layer8_bias = Alex_Layer8_bias_data;

	*Alex_Layer1_Weights = Alex_Layer1_Weights_data;
	*Alex_Layer2_Weights = Alex_Layer2_Weights_data;
	*Alex_Layer3_Weights = Alex_Layer3_Weights_data;
	*Alex_Layer4_Weights = Alex_Layer4_Weights_data;
	*Alex_Layer5_Weights = Alex_Layer5_Weights_data;
	*Alex_Layer6_Weights = Alex_Layer6_Weights_data;
	*Alex_Layer7_Weights = Alex_Layer7_Weights_data;
	*Alex_Layer8_Weights = Alex_Layer8_Weights_data;

	free(Alex_Layer1_Neurons_CPU);

	free(Alex_Layer1_bias_CPU);
	free(Alex_Layer2_bias_CPU);
	free(Alex_Layer3_bias_CPU);
	free(Alex_Layer4_bias_CPU);
	free(Alex_Layer5_bias_CPU);
	free(Alex_Layer6_bias_CPU);
	free(Alex_Layer7_bias_CPU);
	free(Alex_Layer8_bias_CPU);

    free(Alex_Layer1_Weights_CPU);
    free(Alex_Layer2_Weights_CPU);
    free(Alex_Layer3_Weights_CPU);
    free(Alex_Layer4_Weights_CPU);
    free(Alex_Layer5_Weights_CPU);
    free(Alex_Layer6_Weights_CPU);
    free(Alex_Layer7_Weights_CPU);
    free(Alex_Layer8_Weights_CPU);

	float *Alex_Layer1_norm_data; 
	cudaMalloc((void**) &Alex_Layer1_norm_data, (64*55*55) * sizeof(float)); //64*55*55 
	*Alex_Layer1_norm = Alex_Layer1_norm_data;

	float *Alex_Layer1_pool_data;
    cudaMalloc((void**) &Alex_Layer1_pool_data, (64*55*55) * sizeof(float)); //64*55*55
	*Alex_Layer1_pool = Alex_Layer1_pool_data;

	float *Alex_Layer2_Neurons_data;
	cudaMalloc((void**) &Alex_Layer2_Neurons_data, (64*27*27) * sizeof(float)); //64*27*27
	*Alex_Layer2_Neurons = Alex_Layer2_Neurons_data;

	float *Alex_Layer2_norm_data;
	cudaMalloc((void**) &Alex_Layer2_norm_data, (192*27*27) * sizeof(float)); //192*27*27
	*Alex_Layer2_norm = Alex_Layer2_norm_data;

	float *Alex_Layer2_pool_data;
    cudaMalloc((void**) &Alex_Layer2_pool_data, (192*27*27) * sizeof(float)); //192*27*27
	*Alex_Layer2_pool = Alex_Layer2_pool_data;

	float *Alex_Layer3_Neurons_data;
    cudaMalloc((void**) &Alex_Layer3_Neurons_data, (192*13*13) * sizeof(float)); //192*13*13
	*Alex_Layer3_Neurons = Alex_Layer3_Neurons_data;

	float *Alex_Layer4_Neurons_data;
    cudaMalloc((void**) &Alex_Layer4_Neurons_data, (384*13*13) * sizeof(float)); //384*13*13
	*Alex_Layer4_Neurons = Alex_Layer4_Neurons_data;

	float *Alex_Layer5_Neurons_data;
	cudaMalloc((void**) &Alex_Layer5_Neurons_data, (256*13*13) * sizeof(float)); //256*13*13
	*Alex_Layer5_Neurons = Alex_Layer5_Neurons_data;

	float *Alex_Layer5_pool_data;
	cudaMalloc((void**) &Alex_Layer5_pool_data, (256*13*13) * sizeof(float)); //256*13*13
	*Alex_Layer5_pool = Alex_Layer5_pool_data;

	float *Alex_Layer6_Neurons_data;
	cudaMalloc((void**) &Alex_Layer6_Neurons_data, (256*6*6) * sizeof(float)); //256*6*6
	*Alex_Layer6_Neurons = Alex_Layer6_Neurons_data;

	float *Alex_Layer7_Neurons_data;
	cudaMalloc((void**) &Alex_Layer7_Neurons_data, 4096 * sizeof(float)); //4096
	*Alex_Layer7_Neurons = Alex_Layer7_Neurons_data;

	float *Alex_Layer8_Neurons_data;
	cudaMalloc((void**) &Alex_Layer8_Neurons_data, 4096 * sizeof(float)); //4096
	*Alex_Layer8_Neurons = Alex_Layer8_Neurons_data;

	float *Alex_Result_Neurons_data;
	cudaMalloc((void**) &Alex_Result_Neurons_data, 1000 * sizeof(float)); //1000
	*Alex_Result_Neurons = Alex_Result_Neurons_data;
}

void inference_alexnet(float *Alex_Layer1_Neurons,float *Alex_Layer2_Neurons,float *Alex_Layer3_Neurons,float *Alex_Layer4_Neurons,
					float *Alex_Layer5_Neurons,float *Alex_Layer6_Neurons,float *Alex_Layer7_Neurons,float *Alex_Layer8_Neurons,
                    float *Alex_Layer1_bias,float *Alex_Layer2_bias,float *Alex_Layer3_bias,float *Alex_Layer4_bias,
                    float *Alex_Layer5_bias,float *Alex_Layer6_bias,float *Alex_Layer7_bias,float *Alex_Layer8_bias,
                    float *Alex_Layer1_Weights,float *Alex_Layer2_Weights,float *Alex_Layer3_Weights,float *Alex_Layer4_Weights,
                    float *Alex_Layer5_Weights,float * Alex_Layer6_Weights,float *Alex_Layer7_Weights,float *Alex_Layer8_Weights,
                    float *Alex_Layer1_pool,float *Alex_Layer2_pool,float *Alex_Layer5_pool,
					float *Alex_Layer1_norm,float *Alex_Layer2_norm,float *Alex_Result_Neurons)
{
	//* First Layer *//
	dim3 Layer1_Block(64,5,5);
	dim3 Layer1_Thread(11,11);
	first<<<Layer1_Block,Layer1_Thread>>>(Alex_Layer1_bias,Alex_Layer1_Neurons,Alex_Layer1_Weights,Alex_Layer1_norm,224,55,4,2,11,3,true,true);

	/* Normalization of First Layer */
	dim3 Norm11_Block(64,5,5);
	dim3 Norm11_Thread(11,11);
	norm<<<Norm11_Block,Norm11_Thread>>>(Alex_Layer1_norm,Alex_Layer1_pool,0.0001,0.75,5,55);

	/* Maxpooling of First Layer */
	dim3 Pool1_Block(64,1,1);
	dim3 Pool1_Thread(27,27);
	max<<<Pool1_Block,Pool1_Thread>>>(Alex_Layer1_pool,Alex_Layer2_Neurons,55,27,2,0,3);

	//* Second Layer *//
	/* Convolution of Second Layer */
	dim3 Layer2_Block(192,1,1);
	dim3 Layer2_Thread(27,27); 
	conv<<<Layer2_Block,Layer2_Thread>>>(Alex_Layer2_bias,Alex_Layer2_Neurons,Alex_Layer2_Weights,Alex_Layer2_norm,27,27,1,2,5,64,true,true);
	
	/* Normalization of Second Layer */
	dim3 Norm2_Block(192,1,1);
	dim3 Norm2_Thread(27,27);
	norm<<<Norm2_Block,Norm2_Thread>>>(Alex_Layer2_norm,Alex_Layer2_pool,0.0001,0.75,5,27);
	
	/* Maxpooling of Second Layer */
	dim3 Pool2_Block(192,1,1);
	dim3 Pool2_Thread(13,13);
	max<<<Pool2_Block,Pool2_Thread>>>(Alex_Layer2_pool,Alex_Layer3_Neurons,27,13,2,0,3);

	//* Third Layer *//
	/* Convolution of Third Layer */
	dim3 Layer3_Block(384,1,1);
	dim3 Layer3_Thread(13,13); 
	conv<<<Layer3_Block,Layer3_Thread>>>(Alex_Layer3_bias, Alex_Layer3_Neurons, Alex_Layer3_Weights, Alex_Layer4_Neurons,13,13,1,1,3,192,true,true);

	//* Fourth Layer *//
	/* Convolution of Fourth Layer */
	dim3 Layer4_Block(256,1,1);
	dim3 Layer4_Thread(13,13); 
	conv<<<Layer4_Block,Layer4_Thread>>>(Alex_Layer4_bias, Alex_Layer4_Neurons, Alex_Layer4_Weights, Alex_Layer5_Neurons,13,13,1,1,3,384,true,true);

	//* Fifth Layer *//
	/* Convolution of Fifth Layer */
	dim3 Layer5_Block(256,1,1);
	dim3 Layer5_Thread(13,13); 
	conv<<<Layer5_Block,Layer5_Thread>>>(Alex_Layer5_bias, Alex_Layer5_Neurons, Alex_Layer5_Weights, Alex_Layer5_pool,13,13,1,1,3,256,true,true);

	/* Maxpooling of Fifth Layer */
	dim3 Pool3_Block(256,1,1);
	dim3 Pool3_Thread(6,6);
	max<<<Pool3_Block,Pool3_Thread>>>(Alex_Layer5_pool, Alex_Layer6_Neurons,13,6,2,0,3);

	//* Sixth Layer *//
	/* First Fully Connected Layer */
	dim3 Layer6_Block(4096,1,1);
	dim3 Layer6_Thread(1,1);
	fc<<<Layer6_Block,Layer6_Thread>>>(Alex_Layer6_bias, Alex_Layer6_Neurons, Alex_Layer6_Weights, Alex_Layer7_Neurons, (6*6*256), true);

	//* Seventh Layer *//
	/* Second Fully Connected Layer */
	dim3 Layer7_Block(4096,1,1);
	dim3 Layer7_Thread(1,1);
	fc<<<Layer7_Block,Layer7_Thread>>>(Alex_Layer7_bias, Alex_Layer7_Neurons, Alex_Layer7_Weights, Alex_Layer8_Neurons, 4096, true);

	//* Eighth Layer *//
	/* Third Fully Connected Layer */
	dim3 Layer8_Block(1000,1,1);
	dim3 Layer8_Thread(1,1);
	fc<<<Layer8_Block,Layer8_Thread>>>(Alex_Layer8_bias, Alex_Layer8_Neurons, Alex_Layer8_Weights, Alex_Result_Neurons, 4096, false);

	float *Alex_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
	cudaMemcpy(Alex_Result_Neurons_CPU, Alex_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

	float max1 = 0.0;
	int index1 = 0; 
	for(int i = 0; i < 1000; i++){
		if(max1 < Alex_Result_Neurons_CPU[i]){
			max1 = Alex_Result_Neurons_CPU[i];	
			index1 = i;
		}
	}
	
	int line_count1 = 0;
	char buffer[1000];
	FILE *list1 = fopen("common/imagenet1000_clsidx_to_labels.txt","rt");
	while(fgets(buffer, 1000, list1) != NULL){
		line_count1++;
		if(line_count1 == (index1+1)){
			printf("Alexnet: %d, %s", index1, buffer);
			break;
		}
	}
	fclose(list1);
	
	free(Alex_Result_Neurons_CPU);
}

void free_alexnet(float *Alex_Layer1_Neurons,float *Alex_Layer2_Neurons,float *Alex_Layer3_Neurons,float *Alex_Layer4_Neurons,
					float *Alex_Layer5_Neurons,float *Alex_Layer6_Neurons,float *Alex_Layer7_Neurons,float *Alex_Layer8_Neurons,
                    float *Alex_Layer1_bias,float *Alex_Layer2_bias,float *Alex_Layer3_bias,float *Alex_Layer4_bias,
                    float *Alex_Layer5_bias,float *Alex_Layer6_bias,float *Alex_Layer7_bias,float *Alex_Layer8_bias,
                    float *Alex_Layer1_Weights,float *Alex_Layer2_Weights,float *Alex_Layer3_Weights,float *Alex_Layer4_Weights,
                    float *Alex_Layer5_Weights,float * Alex_Layer6_Weights,float *Alex_Layer7_Weights,float *Alex_Layer8_Weights,
                    float *Alex_Layer1_pool,float *Alex_Layer2_pool,float *Alex_Layer5_pool,
					float *Alex_Layer1_norm,float *Alex_Layer2_norm,float *Alex_Result_Neurons)
{
	cudaFree(Alex_Layer1_Neurons);
	cudaFree(Alex_Layer2_Neurons);
	cudaFree(Alex_Layer3_Neurons);
	cudaFree(Alex_Layer4_Neurons);
	cudaFree(Alex_Layer5_Neurons);
	cudaFree(Alex_Layer6_Neurons);
	cudaFree(Alex_Layer7_Neurons);
	cudaFree(Alex_Layer8_Neurons);

	cudaFree(Alex_Layer1_bias);
	cudaFree(Alex_Layer2_bias);
	cudaFree(Alex_Layer3_bias);
	cudaFree(Alex_Layer4_bias);
	cudaFree(Alex_Layer5_bias);
	cudaFree(Alex_Layer6_bias);
	cudaFree(Alex_Layer7_bias);
	cudaFree(Alex_Layer8_bias);

	cudaFree(Alex_Layer1_Weights);
	cudaFree(Alex_Layer2_Weights);
	cudaFree(Alex_Layer3_Weights);
	cudaFree(Alex_Layer4_Weights);
	cudaFree(Alex_Layer5_Weights);
	cudaFree(Alex_Layer6_Weights);
	cudaFree(Alex_Layer7_Weights);
	cudaFree(Alex_Layer8_Weights);

	cudaFree(Alex_Layer1_pool);
	cudaFree(Alex_Layer2_pool);
	cudaFree(Alex_Layer5_pool);
	cudaFree(Alex_Layer1_norm);
	cudaFree(Alex_Layer2_norm);
	cudaFree(Alex_Result_Neurons);
}

int main(void) {

	float *Alex_Layer1_Neurons; float *Alex_Layer2_Neurons; float *Alex_Layer3_Neurons; float *Alex_Layer4_Neurons; 
	float *Alex_Layer5_Neurons; float *Alex_Layer6_Neurons; float *Alex_Layer7_Neurons; float *Alex_Layer8_Neurons; 
	float *Alex_Layer1_bias; float *Alex_Layer2_bias; float *Alex_Layer3_bias; float *Alex_Layer4_bias; 
	float *Alex_Layer5_bias; float *Alex_Layer6_bias; float *Alex_Layer7_bias; float *Alex_Layer8_bias; 
	float *Alex_Layer1_Weights; float *Alex_Layer2_Weights; float *Alex_Layer3_Weights; float *Alex_Layer4_Weights; 
	float *Alex_Layer5_Weights; float *Alex_Layer6_Weights; float *Alex_Layer7_Weights; float *Alex_Layer8_Weights; 
	float *Alex_Layer1_pool; float *Alex_Layer2_pool; float *Alex_Layer5_pool; 
	float *Alex_Layer1_norm; float *Alex_Layer2_norm; float *Alex_Result_Neurons;

	printf("Copy weights from CPU to GPU\n");
	host2gpu_alexnet(&Alex_Layer1_Neurons, &Alex_Layer2_Neurons, &Alex_Layer3_Neurons, &Alex_Layer4_Neurons,
		&Alex_Layer5_Neurons, &Alex_Layer6_Neurons, &Alex_Layer7_Neurons, &Alex_Layer8_Neurons,
		&Alex_Layer1_bias, &Alex_Layer2_bias, &Alex_Layer3_bias, &Alex_Layer4_bias,
		&Alex_Layer5_bias, &Alex_Layer6_bias, &Alex_Layer7_bias, &Alex_Layer8_bias,
		&Alex_Layer1_Weights, &Alex_Layer2_Weights, &Alex_Layer3_Weights, &Alex_Layer4_Weights,
		&Alex_Layer5_Weights, &Alex_Layer6_Weights, &Alex_Layer7_Weights, &Alex_Layer8_Weights,
		&Alex_Layer1_pool, &Alex_Layer2_pool, &Alex_Layer5_pool,
		&Alex_Layer1_norm, &Alex_Layer2_norm, &Alex_Result_Neurons);

	printf("Inference start\n");
	inference_alexnet(Alex_Layer1_Neurons, Alex_Layer2_Neurons, Alex_Layer3_Neurons, Alex_Layer4_Neurons,
		Alex_Layer5_Neurons, Alex_Layer6_Neurons, Alex_Layer7_Neurons, Alex_Layer8_Neurons,
		Alex_Layer1_bias, Alex_Layer2_bias, Alex_Layer3_bias, Alex_Layer4_bias,
		Alex_Layer5_bias, Alex_Layer6_bias, Alex_Layer7_bias, Alex_Layer8_bias,
		Alex_Layer1_Weights, Alex_Layer2_Weights, Alex_Layer3_Weights, Alex_Layer4_Weights,
		Alex_Layer5_Weights,  Alex_Layer6_Weights, Alex_Layer7_Weights, Alex_Layer8_Weights,
		Alex_Layer1_pool, Alex_Layer2_pool, Alex_Layer5_pool,
		Alex_Layer1_norm, Alex_Layer2_norm, Alex_Result_Neurons);

	printf("Free memory\n");
	free_alexnet(Alex_Layer1_Neurons, Alex_Layer2_Neurons, Alex_Layer3_Neurons, Alex_Layer4_Neurons,
		Alex_Layer5_Neurons, Alex_Layer6_Neurons, Alex_Layer7_Neurons, Alex_Layer8_Neurons,
		Alex_Layer1_bias, Alex_Layer2_bias, Alex_Layer3_bias, Alex_Layer4_bias,
		Alex_Layer5_bias, Alex_Layer6_bias, Alex_Layer7_bias, Alex_Layer8_bias,
		Alex_Layer1_Weights, Alex_Layer2_Weights, Alex_Layer3_Weights, Alex_Layer4_Weights,
		Alex_Layer5_Weights,  Alex_Layer6_Weights, Alex_Layer7_Weights, Alex_Layer8_Weights,
		Alex_Layer1_pool, Alex_Layer2_pool, Alex_Layer5_pool,
		Alex_Layer1_norm, Alex_Layer2_norm, Alex_Result_Neurons);
}