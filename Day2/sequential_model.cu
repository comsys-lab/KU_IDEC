#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "common/Kernel.cu"

#define INPUT_SIZE 224*224*3


/* Function to Read Alexnet Input Parameters */
extern "C"{
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

void alex_first_conv(float *Alex_Layer1_bias,float *Alex_Layer1_Neurons,float *Alex_Layer1_Weights,float *Alex_Layer1_norm)
{
    dim3 Layer1_Block(64,5,5);
	dim3 Layer1_Thread(11,11);
	first<<<Layer1_Block,Layer1_Thread>>>(Alex_Layer1_bias,Alex_Layer1_Neurons,Alex_Layer1_Weights,Alex_Layer1_norm,224,55,4,2,11,3,true,true);
}

void alex_fisrt_norm(float *Alex_Layer1_norm,float *Alex_Layer1_pool)
{
   	dim3 Norm11_Block(64,5,5);
	dim3 Norm11_Thread(11,11);
	norm<<<Norm11_Block,Norm11_Thread>>>(Alex_Layer1_norm,Alex_Layer1_pool,0.0001,0.75,5,55); 
}

void alex_first_pool(float *Alex_Layer1_pool,float *Alex_Layer2_Neurons)
{
    dim3 Pool1_Block(64,1,1);
	dim3 Pool1_Thread(27,27);
	max<<<Pool1_Block,Pool1_Thread>>>(Alex_Layer1_pool,Alex_Layer2_Neurons,55,27,2,0,3);
}

void alex_second_conv(float *Alex_Layer2_bias,float *Alex_Layer2_Neurons,float *Alex_Layer2_Weights,float *Alex_Layer2_norm)
{
    dim3 Layer2_Block(192,1,1);
	dim3 Layer2_Thread(27,27); 
	AlexNet_conv<<<Layer2_Block,Layer2_Thread>>>(Alex_Layer2_bias,Alex_Layer2_Neurons,Alex_Layer2_Weights,Alex_Layer2_norm,27,27,1,2,5,64,true,true);
}

void alex_second_norm(float *Alex_Layer2_norm,float *Alex_Layer2_pool)
{
    dim3 Norm2_Block(192,1,1);
	dim3 Norm2_Thread(27,27);
	norm<<<Norm2_Block,Norm2_Thread>>>(Alex_Layer2_norm,Alex_Layer2_pool,0.0001,0.75,5,27);
}

void alex_second_pool(float *Alex_Layer2_pool,float *Alex_Layer3_Neurons)
{
    dim3 Pool2_Block(192,1,1);
	dim3 Pool2_Thread(13,13);
	max<<<Pool2_Block,Pool2_Thread>>>(Alex_Layer2_pool,Alex_Layer3_Neurons,27,13,2,0,3);
}

void alex_third_conv(float *Alex_Layer3_bias,float *Alex_Layer3_Neurons,float *Alex_Layer3_Weights,float *Alex_Layer4_Neurons)
{
	dim3 Layer3_Block(384,1,1);
	dim3 Layer3_Thread(13,13); 
	AlexNet_conv<<<Layer3_Block,Layer3_Thread>>>(Alex_Layer3_bias,Alex_Layer3_Neurons,Alex_Layer3_Weights,Alex_Layer4_Neurons,13,13,1,1,3,192,true,true);
}

void alex_fourth_conv(float *Alex_Layer4_bias,float *Alex_Layer4_Neurons,float *Alex_Layer4_Weights,float *Alex_Layer5_Neurons)
{
    dim3 Layer4_Block(256,1,1);
	dim3 Layer4_Thread(13,13); 
	AlexNet_conv<<<Layer4_Block,Layer4_Thread>>>(Alex_Layer4_bias,Alex_Layer4_Neurons,Alex_Layer4_Weights,Alex_Layer5_Neurons,13,13,1,1,3,384,true,true);
}

void alex_fifth_conv(float *Alex_Layer5_bias,float *Alex_Layer5_Neurons,float *Alex_Layer5_Weights,float *Alex_Layer5_pool)
{
    dim3 Layer5_Block(256,1,1);
	dim3 Layer5_Thread(13,13); 
	AlexNet_conv<<<Layer5_Block,Layer5_Thread>>>(Alex_Layer5_bias,Alex_Layer5_Neurons,Alex_Layer5_Weights,Alex_Layer5_pool,13,13,1,1,3,256,true,true);
}

void alex_fifth_pool(float *Alex_Layer5_pool,float *Alex_Layer6_Neurons)
{
	dim3 Pool3_Block(256,1,1);
	dim3 Pool3_Thread(6,6);
	max<<<Pool3_Block,Pool3_Thread>>>(Alex_Layer5_pool,Alex_Layer6_Neurons,13,6,2,0,3);
}

void alex_first_fc(float *Alex_Layer6_bias,float *Alex_Layer6_Neurons,float *Alex_Layer6_Weights,float *Alex_Layer7_Neurons)
{	
	dim3 Layer6_Block(4096,1,1);
	dim3 Layer6_Thread(1,1);
	AlexNet_fc<<<Layer6_Block,Layer6_Thread>>>(Alex_Layer6_bias,Alex_Layer6_Neurons,Alex_Layer6_Weights,Alex_Layer7_Neurons,(6*6*256),true);

}

void alex_second_fc(float *Alex_Layer7_bias,float *Alex_Layer7_Neurons,float *Alex_Layer7_Weights,float *Alex_Layer8_Neurons)
{
	dim3 Layer7_Block(4096,1,1);
	dim3 Layer7_Thread(1,1);
	AlexNet_fc<<<Layer7_Block,Layer7_Thread>>>(Alex_Layer7_bias,Alex_Layer7_Neurons,Alex_Layer7_Weights,Alex_Layer8_Neurons,4096,true);
}

void alex_third_fc(float *Alex_Layer8_bias,float *Alex_Layer8_Neurons,float *Alex_Layer8_Weights,float *Alex_Result_Neurons)
{


    dim3 Layer8_Block(1000,1,1);
	dim3 Layer8_Thread(1,1);
	AlexNet_fc<<<Layer8_Block,Layer8_Thread>>>(Alex_Layer8_bias,Alex_Layer8_Neurons,Alex_Layer8_Weights,Alex_Result_Neurons,4096,false);

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
			// printf("\n---Alexnet Result---");
			// printf("\nClass ID: %d\nClass Name: %sProbability: %f\n", index1, buffer, max1);
			printf("\nAlexnet: %d, %s", index1, buffer);
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

void host2gpu_resnet18(float **Res_Layer1_Neurons,float **Res_Layer2_Neurons,float **Res_Layer3_Neurons,float **Res_Layer4_Neurons,
					float **Res_Layer5_Neurons,float **Res_Layer6_Neurons,float **Res_Layer7_Neurons,float **Res_Layer8_Neurons,
					float **Res_Layer9_Neurons,float **Res_Layer10_Neurons,float **Res_Layer11_Neurons,float **Res_Layer12_Neurons,
					float **Res_Layer13_Neurons,float **Res_Layer14_Neurons,float **Res_Layer15_Neurons,float **Res_Layer16_Neurons,
					float **Res_Layer17_Neurons,float **Res_Layer18_Neurons,
                    float **Res_Layer1_Weights,float **Res_Layer2_Weights,float **Res_Layer3_Weights,float **Res_Layer4_Weights,
                    float **Res_Layer5_Weights,float **Res_Layer6_Weights,float **Res_Layer7_Weights,float **Res_Layer8_Weights,
                    float **Res_Layer9_Weights,float **Res_Layer10_Weights,float **Res_Layer11_Weights,float **Res_Layer12_Weights,
                    float **Res_Layer13_Weights,float **Res_Layer14_Weights,float **Res_Layer15_Weights,float **Res_Layer16_Weights,
                    float **Res_Layer17_Weights,float **Res_Block3_Weights,float **Res_Block4_Weights,float **Res_Block5_Weights,
                    float **Res_Layer1_Gamma,float **Res_Layer2_Gamma,float **Res_Layer3_Gamma,float **Res_Layer4_Gamma,
                    float **Res_Layer5_Gamma,float **Res_Layer6_Gamma,float **Res_Layer7_Gamma,float **Res_Layer8_Gamma,
                    float **Res_Layer9_Gamma,float **Res_Layer10_Gamma,float **Res_Layer11_Gamma,float **Res_Layer12_Gamma,
                    float **Res_Layer13_Gamma,float **Res_Layer14_Gamma,float **Res_Layer15_Gamma,float **Res_Layer16_Gamma,
                    float **Res_Layer17_Gamma,float **Res_Block3_Gamma,float **Res_Block4_Gamma,float **Res_Block5_Gamma,
                    float **Res_Layer1_Beta,float **Res_Layer2_Beta,float**Res_Layer3_Beta,float **Res_Layer4_Beta,
                    float **Res_Layer5_Beta,float **Res_Layer6_Beta,float **Res_Layer7_Beta,float **Res_Layer8_Beta,
                    float **Res_Layer9_Beta,float **Res_Layer10_Beta,float **Res_Layer11_Beta,float **Res_Layer12_Beta,
                    float **Res_Layer13_Beta,float **Res_Layer14_Beta,float **Res_Layer15_Beta,float **Res_Layer16_Beta,
                    float **Res_Layer17_Beta,float **Res_Block3_Beta,float **Res_Block4_Beta,float **Res_Block5_Beta,
                    float **Res_mean1,float **Res_mean2,float **Res_mean3,float **Res_mean4,float **Res_mean5,
                    float **Res_mean6,float **Res_mean7,float **Res_mean8,float **Res_mean9,float **Res_mean10,
                    float **Res_mean11,float **Res_mean12,float **Res_mean13,float **Res_mean14,float **Res_mean15,
                    float **Res_mean16,float **Res_mean17,float **Res_Block3_mean,float **Res_Block4_mean,float **Res_Block5_mean,
                    float **Res_var1,float **Res_var2,float **Res_var3,float **Res_var4,float **Res_var5,
                    float **Res_var6,float **Res_var7,float **Res_var8,float **Res_var9,float **Res_var10,
                    float **Res_var11,float **Res_var12,float **Res_var13,float **Res_var14,float **Res_var15,
                    float **Res_var16,float **Res_var17,float **Res_Block3_var,float **Res_Block4_var,float **Res_Block5_var,
                    float **Res_FC_bias,float **Res_FC_Weights,
					float **Res_Layer3_basic,float **Res_Layer5_basic,float **Res_Layer7_basic,float **Res_Layer9_basic,
					float **Res_Layer11_basic,float **Res_Layer13_basic,float **Res_Layer15_basic,float **Res_Layer17_basic,
					float **Res_Block3_basic,float **Res_Block4_basic,float **Res_Block5_basic,
					float **Res_Layer1_bn,float **Res_Layer2_bn,float **Res_Layer3_bn,float **Res_Layer4_bn,
					float **Res_Layer5_bn,float **Res_Layer6_bn,float **Res_Layer7_bn,float **Res_Layer8_bn,
					float **Res_Layer9_bn,float **Res_Layer10_bn,float **Res_Layer11_bn,float **Res_Layer12_bn,
					float **Res_Layer13_bn,float **Res_Layer14_bn,float **Res_Layer15_bn,float **Res_Layer16_bn,
					float **Res_Layer17_bn,float **Res_Block3_bn,float **Res_Block4_bn,float **Res_Block5_bn,
					float **Res_Layer1_pool,float **Res_FC_Neurons,float **Res_Result_Neurons)
{
	float *Res_Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));
	read_parameter("data_resnet18/input_cat.txt", Res_Layer1_Neurons_CPU);

	float *Res_Layer1_Weights_CPU = (float*) malloc ((7*7*3*64) * sizeof(float)); // = 9,408
	float *Res_Layer2_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Res_Layer3_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Res_Layer4_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Res_Layer5_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Res_Layer6_Weights_CPU = (float*) malloc ((3*3*64*128) * sizeof(float)); // = 73,728
	float *Res_Layer7_Weights_CPU = (float*) malloc ((3*3*128*128) * sizeof(float)); // = 147,456
	float *Res_Layer8_Weights_CPU = (float*) malloc ((3*3*128*128) * sizeof(float)); // = 147,456
    float *Res_Layer9_Weights_CPU = (float*) malloc ((3*3*128*128) * sizeof(float)); // = 147,456
	float *Res_Layer10_Weights_CPU = (float*) malloc ((3*3*128*256) * sizeof(float)); // = 294,912
	float *Res_Layer11_Weights_CPU = (float*) malloc ((3*3*256*256) * sizeof(float)); // = 589,824
	float *Res_Layer12_Weights_CPU = (float*) malloc ((3*3*256*256) * sizeof(float)); // = 589,824
	float *Res_Layer13_Weights_CPU = (float*) malloc ((3*3*256*256) * sizeof(float)); // = 589,824
	float *Res_Layer14_Weights_CPU = (float*) malloc ((3*3*256*512) * sizeof(float)); // = 1,179,648
	float *Res_Layer15_Weights_CPU = (float*) malloc ((3*3*512*512) * sizeof(float)); // = 2,359,296
	float *Res_Layer16_Weights_CPU = (float*) malloc ((3*3*512*512) * sizeof(float)); // = 2,359,296
	float *Res_Layer17_Weights_CPU = (float*) malloc ((3*3*512*512) * sizeof(float)); // = 2,359,296
	float *Res_Block3_Weights_CPU = (float*) malloc ((1*1*64*128) * sizeof(float)); // = 8,192
	float *Res_Block4_Weights_CPU = (float*) malloc ((1*1*128*256) * sizeof(float)); // = 32,768
	float *Res_Block5_Weights_CPU = (float*) malloc ((1*1*256*512) * sizeof(float)); // = 131,072
   
    float *Res_Layer1_Gamma_CPU = (float*) malloc (64 * sizeof(float));
	float *Res_Layer2_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer3_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer4_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer5_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer6_Gamma_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_Layer7_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_Layer8_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
    float *Res_Layer9_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_Layer10_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer11_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer12_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer13_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer14_Gamma_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_Layer15_Gamma_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_Layer16_Gamma_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_Layer17_Gamma_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_Block3_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_Block4_Gamma_CPU = (float*) malloc (256 * sizeof(float));
	float *Res_Block5_Gamma_CPU = (float*) malloc (512 * sizeof(float)); 
    
	float *Res_Layer1_Beta_CPU = (float*) malloc (64 * sizeof(float));
	float *Res_Layer2_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer3_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer4_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer5_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_Layer6_Beta_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_Layer7_Beta_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_Layer8_Beta_CPU = (float*) malloc (128 * sizeof(float)); 
    float *Res_Layer9_Beta_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_Layer10_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer11_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer12_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer13_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_Layer14_Beta_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_Layer15_Beta_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_Layer16_Beta_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_Layer17_Beta_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_Block3_Beta_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_Block4_Beta_CPU = (float*) malloc (256 * sizeof(float));
	float *Res_Block5_Beta_CPU = (float*) malloc (512 * sizeof(float));
   
	float *Res_mean1_CPU = (float*) malloc (64 * sizeof(float));
	float *Res_mean2_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_mean3_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_mean4_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_mean5_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_mean6_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_mean7_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_mean8_CPU = (float*) malloc (128 * sizeof(float)); 
    float *Res_mean9_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_mean10_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_mean11_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_mean12_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_mean13_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_mean14_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_mean15_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_mean16_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_mean17_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_Block3_mean_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_Block4_mean_CPU = (float*) malloc (256 * sizeof(float));
	float *Res_Block5_mean_CPU = (float*) malloc (512 * sizeof(float));
   
	float *Res_var1_CPU = (float*) malloc (64 * sizeof(float));
	float *Res_var2_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_var3_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_var4_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_var5_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Res_var6_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_var7_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_var8_CPU = (float*) malloc (128 * sizeof(float)); 
    float *Res_var9_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Res_var10_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_var11_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_var12_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_var13_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Res_var14_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_var15_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_var16_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Res_var17_CPU = (float*) malloc (512 * sizeof(float));
	float *Res_Block3_var_CPU = (float*) malloc (128 * sizeof(float));
	float *Res_Block4_var_CPU = (float*) malloc (256 * sizeof(float));
	float *Res_Block5_var_CPU = (float*) malloc (512 * sizeof(float));
   
	float *Res_FC_bias_CPU = (float*) malloc (1000* sizeof(float));
	float *Res_FC_Weights_CPU = (float*) malloc ((512*1000) * sizeof(float));

	read_parameter("data_resnet18/conv_data/conv1.txt", Res_Layer1_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv2.txt", Res_Layer2_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv3.txt", Res_Layer3_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv4.txt", Res_Layer4_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv5.txt", Res_Layer5_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv6.txt", Res_Layer6_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv7.txt", Res_Layer7_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv8.txt", Res_Layer8_Weights_CPU);
 	read_parameter("data_resnet18/conv_data/conv9.txt", Res_Layer9_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv10.txt", Res_Layer10_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv11.txt", Res_Layer11_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv12.txt", Res_Layer12_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv13.txt", Res_Layer13_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv14.txt", Res_Layer14_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv15.txt", Res_Layer15_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv16.txt", Res_Layer16_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv17.txt", Res_Layer17_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv_block3.txt", Res_Block3_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv_block4.txt", Res_Block4_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv_block5.txt", Res_Block5_Weights_CPU);

	read_parameter("data_resnet18/gamma_data/gamma1.txt", Res_Layer1_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma2.txt", Res_Layer2_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma3.txt", Res_Layer3_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma4.txt", Res_Layer4_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma5.txt", Res_Layer5_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma6.txt", Res_Layer6_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma7.txt", Res_Layer7_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma8.txt", Res_Layer8_Gamma_CPU);
 	read_parameter("data_resnet18/gamma_data/gamma9.txt", Res_Layer9_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma10.txt", Res_Layer10_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma11.txt", Res_Layer11_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma12.txt", Res_Layer12_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma13.txt", Res_Layer13_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma14.txt", Res_Layer14_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma15.txt", Res_Layer15_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma16.txt", Res_Layer16_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma17.txt", Res_Layer17_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma_block3.txt", Res_Block3_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma_block4.txt", Res_Block4_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma_block5.txt", Res_Block5_Gamma_CPU);

	read_parameter("data_resnet18/beta_data/beta1.txt", Res_Layer1_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta2.txt", Res_Layer2_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta3.txt", Res_Layer3_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta4.txt", Res_Layer4_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta5.txt", Res_Layer5_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta6.txt", Res_Layer6_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta7.txt", Res_Layer7_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta8.txt", Res_Layer8_Beta_CPU);
 	read_parameter("data_resnet18/beta_data/beta9.txt", Res_Layer9_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta10.txt", Res_Layer10_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta11.txt", Res_Layer11_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta12.txt", Res_Layer12_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta13.txt", Res_Layer13_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta14.txt", Res_Layer14_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta15.txt", Res_Layer15_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta16.txt", Res_Layer16_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta17.txt", Res_Layer17_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta_block3.txt", Res_Block3_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta_block4.txt", Res_Block4_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta_block5.txt", Res_Block5_Beta_CPU);

	read_parameter("data_resnet18/mean_data/mean1.txt", Res_mean1_CPU);
	read_parameter("data_resnet18/mean_data/mean2.txt", Res_mean2_CPU);
	read_parameter("data_resnet18/mean_data/mean3.txt", Res_mean3_CPU);
	read_parameter("data_resnet18/mean_data/mean4.txt", Res_mean4_CPU);
	read_parameter("data_resnet18/mean_data/mean5.txt", Res_mean5_CPU);
	read_parameter("data_resnet18/mean_data/mean6.txt", Res_mean6_CPU);
	read_parameter("data_resnet18/mean_data/mean7.txt", Res_mean7_CPU);
	read_parameter("data_resnet18/mean_data/mean8.txt", Res_mean8_CPU);
 	read_parameter("data_resnet18/mean_data/mean9.txt", Res_mean9_CPU);
	read_parameter("data_resnet18/mean_data/mean10.txt", Res_mean10_CPU);
	read_parameter("data_resnet18/mean_data/mean11.txt", Res_mean11_CPU);
	read_parameter("data_resnet18/mean_data/mean12.txt", Res_mean12_CPU);
	read_parameter("data_resnet18/mean_data/mean13.txt", Res_mean13_CPU);
	read_parameter("data_resnet18/mean_data/mean14.txt", Res_mean14_CPU);
	read_parameter("data_resnet18/mean_data/mean15.txt", Res_mean15_CPU);
	read_parameter("data_resnet18/mean_data/mean16.txt", Res_mean16_CPU);
	read_parameter("data_resnet18/mean_data/mean17.txt", Res_mean17_CPU);
	read_parameter("data_resnet18/mean_data/mean_block3.txt", Res_Block3_mean_CPU);
	read_parameter("data_resnet18/mean_data/mean_block4.txt", Res_Block4_mean_CPU);
	read_parameter("data_resnet18/mean_data/mean_block5.txt", Res_Block5_mean_CPU);

	read_parameter("data_resnet18/var_data/var1.txt", Res_var1_CPU);
	read_parameter("data_resnet18/var_data/var2.txt", Res_var2_CPU);
	read_parameter("data_resnet18/var_data/var3.txt", Res_var3_CPU);
	read_parameter("data_resnet18/var_data/var4.txt", Res_var4_CPU);
	read_parameter("data_resnet18/var_data/var5.txt", Res_var5_CPU);
	read_parameter("data_resnet18/var_data/var6.txt", Res_var6_CPU);
	read_parameter("data_resnet18/var_data/var7.txt", Res_var7_CPU);
	read_parameter("data_resnet18/var_data/var8.txt", Res_var8_CPU);
 	read_parameter("data_resnet18/var_data/var9.txt", Res_var9_CPU);
	read_parameter("data_resnet18/var_data/var10.txt", Res_var10_CPU);
	read_parameter("data_resnet18/var_data/var11.txt", Res_var11_CPU);
	read_parameter("data_resnet18/var_data/var12.txt", Res_var12_CPU);
	read_parameter("data_resnet18/var_data/var13.txt", Res_var13_CPU);
	read_parameter("data_resnet18/var_data/var14.txt", Res_var14_CPU);
	read_parameter("data_resnet18/var_data/var15.txt", Res_var15_CPU);
	read_parameter("data_resnet18/var_data/var16.txt", Res_var16_CPU);
	read_parameter("data_resnet18/var_data/var17.txt", Res_var17_CPU);
	read_parameter("data_resnet18/var_data/var_block3.txt", Res_Block3_var_CPU);
	read_parameter("data_resnet18/var_data/var_block4.txt", Res_Block4_var_CPU);
	read_parameter("data_resnet18/var_data/var_block5.txt", Res_Block5_var_CPU);

	read_parameter("data_resnet18/fc_data/fc1_bias.txt", Res_FC_bias_CPU);
	read_parameter("data_resnet18/fc_data/fc1_weight.txt", Res_FC_Weights_CPU);

    float *Res_Layer1_Neurons_data;
	float *Res_Layer1_Weights_data, *Res_Layer2_Weights_data, *Res_Layer3_Weights_data, *Res_Layer4_Weights_data, 
			*Res_Layer5_Weights_data, *Res_Layer6_Weights_data, *Res_Layer7_Weights_data, *Res_Layer8_Weights_data, 
			*Res_Layer9_Weights_data, *Res_Layer10_Weights_data, *Res_Layer11_Weights_data, *Res_Layer12_Weights_data, 
			*Res_Layer13_Weights_data, *Res_Layer14_Weights_data, *Res_Layer15_Weights_data, *Res_Layer16_Weights_data, 
			*Res_Layer17_Weights_data, *Res_Block3_Weights_data, *Res_Block4_Weights_data, *Res_Block5_Weights_data; 
	float *Res_Layer1_Gamma_data, *Res_Layer2_Gamma_data, *Res_Layer3_Gamma_data, *Res_Layer4_Gamma_data,
			*Res_Layer5_Gamma_data, *Res_Layer6_Gamma_data, *Res_Layer7_Gamma_data, *Res_Layer8_Gamma_data,
			*Res_Layer9_Gamma_data, *Res_Layer10_Gamma_data, *Res_Layer11_Gamma_data, *Res_Layer12_Gamma_data,
			*Res_Layer13_Gamma_data, *Res_Layer14_Gamma_data, *Res_Layer15_Gamma_data, *Res_Layer16_Gamma_data,
			*Res_Layer17_Gamma_data, *Res_Block3_Gamma_data, *Res_Block4_Gamma_data, *Res_Block5_Gamma_data;
	float *Res_Layer1_Beta_data, *Res_Layer2_Beta_data, *Res_Layer3_Beta_data, *Res_Layer4_Beta_data,
			*Res_Layer5_Beta_data, *Res_Layer6_Beta_data, *Res_Layer7_Beta_data, *Res_Layer8_Beta_data,
			*Res_Layer9_Beta_data, *Res_Layer10_Beta_data, *Res_Layer11_Beta_data, *Res_Layer12_Beta_data,
			*Res_Layer13_Beta_data, *Res_Layer14_Beta_data, *Res_Layer15_Beta_data, *Res_Layer16_Beta_data,
			*Res_Layer17_Beta_data, *Res_Block3_Beta_data, *Res_Block4_Beta_data, *Res_Block5_Beta_data;
	float *Res_mean1_data, *Res_mean2_data, *Res_mean3_data, *Res_mean4_data, *Res_mean5_data,
			*Res_mean6_data, *Res_mean7_data, *Res_mean8_data, *Res_mean9_data, *Res_mean10_data,
			*Res_mean11_data, *Res_mean12_data, *Res_mean13_data, *Res_mean14_data, *Res_mean15_data,
			*Res_mean16_data, *Res_mean17_data, *Res_Block3_mean_data, *Res_Block4_mean_data, *Res_Block5_mean_data;
	float *Res_var1_data, *Res_var2_data, *Res_var3_data, *Res_var4_data, *Res_var5_data,
			*Res_var6_data, *Res_var7_data, *Res_var8_data, *Res_var9_data, *Res_var10_data,
			*Res_var11_data, *Res_var12_data, *Res_var13_data, *Res_var14_data, *Res_var15_data,
			*Res_var16_data, *Res_var17_data, *Res_Block3_var_data, *Res_Block4_var_data, *Res_Block5_var_data;
	float *Res_FC_bias_data, *Res_FC_Weights_data; 

	cudaMalloc((void**) &Res_Layer1_Neurons_data, INPUT_SIZE * sizeof(float)); //224*224*3
	cudaMalloc((void**) &Res_Layer1_Weights_data, sizeof(float) * (7*7*3*64));
	cudaMalloc((void**) &Res_Layer1_Gamma_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer1_Beta_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_mean1_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_var1_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer2_Weights_data, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Res_Layer2_Gamma_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer2_Beta_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_mean2_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_var2_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer3_Weights_data, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Res_Layer3_Gamma_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer3_Beta_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_mean3_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_var3_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer4_Weights_data, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Res_Layer4_Gamma_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer4_Beta_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_mean4_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_var4_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer5_Weights_data, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Res_Layer5_Gamma_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer5_Beta_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_mean5_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_var5_data, sizeof(float) * 64);
	cudaMalloc((void**) &Res_Layer6_Weights_data, sizeof(float) * (3*3*64*128));
	cudaMalloc((void**) &Res_Layer6_Gamma_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer6_Beta_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_mean6_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_var6_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer7_Weights_data, sizeof(float) * (3*3*128*128));
	cudaMalloc((void**) &Res_Layer7_Gamma_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer7_Beta_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_mean7_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_var7_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Block3_Weights_data, sizeof(float) * (1*1*64*128));
	cudaMalloc((void**) &Res_Block3_Gamma_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Block3_Beta_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Block3_mean_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Block3_var_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer8_Weights_data, sizeof(float) * (3*3*128*128));
	cudaMalloc((void**) &Res_Layer8_Gamma_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer8_Beta_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_mean8_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_var8_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer9_Weights_data, sizeof(float) * (3*3*128*128));
	cudaMalloc((void**) &Res_Layer9_Gamma_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer9_Beta_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_mean9_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_var9_data, sizeof(float) * 128);
	cudaMalloc((void**) &Res_Layer10_Weights_data, sizeof(float) * (3*3*128*256));
	cudaMalloc((void**) &Res_Layer10_Gamma_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer10_Beta_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_mean10_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_var10_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer11_Weights_data, sizeof(float) * (3*3*256*256));	
	cudaMalloc((void**) &Res_Layer11_Gamma_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer11_Beta_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_mean11_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_var11_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Block4_Weights_data, sizeof(float) * (1*1*128*256));
	cudaMalloc((void**) &Res_Block4_Gamma_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Block4_Beta_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Block4_mean_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Block4_var_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer12_Weights_data, sizeof(float) * (3*3*256*256));
	cudaMalloc((void**) &Res_Layer12_Gamma_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer12_Beta_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_mean12_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_var12_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer13_Weights_data, sizeof(float) * (3*3*256*256));
	cudaMalloc((void**) &Res_Layer13_Gamma_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer13_Beta_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_mean13_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_var13_data, sizeof(float) * 256);
	cudaMalloc((void**) &Res_Layer14_Weights_data, sizeof(float) * (3*3*256*512));
	cudaMalloc((void**) &Res_Layer14_Gamma_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer14_Beta_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_mean14_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_var14_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer15_Weights_data, sizeof(float) * (3*3*512*512));
	cudaMalloc((void**) &Res_Layer15_Gamma_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer15_Beta_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_mean15_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_var15_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Block5_Weights_data, sizeof(float) * (1*1*256*512));
	cudaMalloc((void**) &Res_Block5_Gamma_data, sizeof(float) * 521);
	cudaMalloc((void**) &Res_Block5_Beta_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Block5_mean_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Block5_var_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer16_Weights_data, sizeof(float) * (3*3*512*512));
	cudaMalloc((void**) &Res_Layer16_Gamma_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer16_Beta_data, sizeof(float) * 512);	
	cudaMalloc((void**) &Res_mean16_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_var16_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer17_Weights_data, sizeof(float) * (3*3*512*512));
	cudaMalloc((void**) &Res_Layer17_Gamma_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_Layer17_Beta_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_mean17_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_var17_data, sizeof(float) * 512);
	cudaMalloc((void**) &Res_FC_bias_data, sizeof(float) * 1000);
	cudaMalloc((void**) &Res_FC_Weights_data, sizeof(float) * (512*1000));

	cudaMemcpy(Res_Layer1_Neurons_data, Res_Layer1_Neurons_CPU, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer1_Weights_data, Res_Layer1_Weights_CPU, sizeof(float) * (7*7*3*64), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer1_Gamma_data, Res_Layer1_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer1_Beta_data, Res_Layer1_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean1_data, Res_mean1_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var1_data, Res_var1_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer2_Weights_data, Res_Layer2_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer2_Gamma_data, Res_Layer2_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer2_Beta_data, Res_Layer2_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean2_data, Res_mean2_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var2_data, Res_var2_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer3_Weights_data, Res_Layer3_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer3_Gamma_data, Res_Layer3_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer3_Beta_data, Res_Layer3_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean3_data, Res_mean3_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var3_data, Res_var3_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer4_Weights_data, Res_Layer4_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);	
	cudaMemcpy(Res_Layer4_Gamma_data, Res_Layer4_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer4_Beta_data, Res_Layer4_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean4_data, Res_mean4_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var4_data, Res_var4_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer5_Weights_data, Res_Layer5_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer5_Gamma_data, Res_Layer5_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer5_Beta_data, Res_Layer5_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean5_data, Res_mean5_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var5_data, Res_var5_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer6_Weights_data, Res_Layer6_Weights_CPU, sizeof(float) * (3*3*64*128), cudaMemcpyHostToDevice);	
	cudaMemcpy(Res_Layer6_Gamma_data, Res_Layer6_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer6_Beta_data, Res_Layer6_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean6_data, Res_mean6_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var6_data, Res_var6_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);	
	cudaMemcpy(Res_Layer7_Weights_data, Res_Layer7_Weights_CPU, sizeof(float) * (3*3*128*128), cudaMemcpyHostToDevice);	
	cudaMemcpy(Res_Layer7_Gamma_data, Res_Layer7_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer7_Beta_data, Res_Layer7_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean7_data, Res_mean7_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var7_data, Res_var7_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block3_Weights_data, Res_Block3_Weights_CPU, sizeof(float) * (1*1*64*128), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block3_Gamma_data, Res_Block3_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block3_Beta_data, Res_Block3_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block3_mean_data, Res_Block3_mean_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block3_var_data, Res_Block3_var_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer8_Weights_data, Res_Layer8_Weights_CPU, sizeof(float) * (3*3*128*128), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer8_Gamma_data, Res_Layer8_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer8_Beta_data, Res_Layer8_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);	
	cudaMemcpy(Res_mean8_data, Res_mean8_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var8_data, Res_var8_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer9_Weights_data, Res_Layer9_Weights_CPU, sizeof(float) * (3*3*128*128), cudaMemcpyHostToDevice);	
	cudaMemcpy(Res_Layer9_Gamma_data, Res_Layer9_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer9_Beta_data, Res_Layer9_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean9_data, Res_mean9_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var9_data, Res_var9_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer10_Weights_data, Res_Layer10_Weights_CPU, sizeof(float) * (3*3*128*256), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer10_Gamma_data, Res_Layer10_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer10_Beta_data, Res_Layer10_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean10_data, Res_mean10_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var10_data, Res_var10_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer11_Weights_data, Res_Layer11_Weights_CPU, sizeof(float) * (3*3*256*256), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer11_Gamma_data, Res_Layer11_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer11_Beta_data, Res_Layer11_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean11_data, Res_mean11_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var11_data, Res_var11_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block4_Weights_data, Res_Block4_Weights_CPU, sizeof(float) * (1*1*128*256), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block4_Gamma_data, Res_Block4_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block4_Beta_data, Res_Block4_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block4_mean_data, Res_Block4_mean_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block4_var_data, Res_Block4_var_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer12_Weights_data, Res_Layer12_Weights_CPU, sizeof(float) * (3*3*256*256), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer12_Gamma_data, Res_Layer12_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer12_Beta_data, Res_Layer12_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean12_data, Res_mean12_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var12_data, Res_var12_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer13_Weights_data, Res_Layer13_Weights_CPU, sizeof(float) * (3*3*256*256), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer13_Gamma_data, Res_Layer13_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer13_Beta_data, Res_Layer13_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean13_data, Res_mean13_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var13_data, Res_var13_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer14_Weights_data, Res_Layer14_Weights_CPU, sizeof(float) * (3*3*256*512), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer14_Gamma_data, Res_Layer14_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer14_Beta_data, Res_Layer14_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean14_data, Res_mean14_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var14_data, Res_var14_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer15_Weights_data, Res_Layer15_Weights_CPU, sizeof(float) * (3*3*512*512), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer15_Gamma_data, Res_Layer15_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer15_Beta_data, Res_Layer15_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean15_data, Res_mean15_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var15_data, Res_var15_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block5_Weights_data, Res_Block5_Weights_CPU, sizeof(float) * (1*1*256*512), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block5_Gamma_data, Res_Block5_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block5_Beta_data, Res_Block5_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block5_mean_data, Res_Block5_mean_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Block5_var_data, Res_Block5_var_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer16_Weights_data, Res_Layer16_Weights_CPU, sizeof(float) * (3*3*512*512), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer16_Gamma_data, Res_Layer16_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer16_Beta_data, Res_Layer16_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean16_data, Res_mean16_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var16_data, Res_var16_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer17_Weights_data, Res_Layer17_Weights_CPU, sizeof(float) * (3*3*512*512), cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer17_Gamma_data, Res_Layer17_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_Layer17_Beta_data, Res_Layer17_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_mean17_data, Res_mean17_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_var17_data, Res_var17_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_FC_bias_data, Res_FC_bias_CPU, sizeof(float) * 1000, cudaMemcpyHostToDevice);
	cudaMemcpy(Res_FC_Weights_data, Res_FC_Weights_CPU, sizeof(float) * (512*1000), cudaMemcpyHostToDevice);

	*Res_Layer1_Neurons = Res_Layer1_Neurons_data;

	*Res_Layer1_Weights = Res_Layer1_Weights_data;
	*Res_Layer2_Weights = Res_Layer2_Weights_data;
	*Res_Layer3_Weights = Res_Layer3_Weights_data;
	*Res_Layer4_Weights = Res_Layer4_Weights_data;
	*Res_Layer5_Weights = Res_Layer5_Weights_data;
	*Res_Layer6_Weights = Res_Layer6_Weights_data;
	*Res_Layer7_Weights = Res_Layer7_Weights_data;
	*Res_Layer8_Weights = Res_Layer8_Weights_data;
	*Res_Layer9_Weights = Res_Layer9_Weights_data;
	*Res_Layer10_Weights = Res_Layer10_Weights_data;
	*Res_Layer11_Weights = Res_Layer11_Weights_data;
	*Res_Layer12_Weights = Res_Layer12_Weights_data;
	*Res_Layer13_Weights = Res_Layer13_Weights_data;
	*Res_Layer14_Weights = Res_Layer14_Weights_data;
	*Res_Layer15_Weights = Res_Layer15_Weights_data;
	*Res_Layer16_Weights = Res_Layer16_Weights_data;
	*Res_Layer17_Weights = Res_Layer17_Weights_data;
	*Res_Block3_Weights = Res_Block3_Weights_data;
	*Res_Block4_Weights = Res_Block4_Weights_data;
	*Res_Block5_Weights = Res_Block5_Weights_data;
	
	*Res_Layer1_Gamma = Res_Layer1_Gamma_data;
	*Res_Layer2_Gamma = Res_Layer2_Gamma_data;
	*Res_Layer3_Gamma = Res_Layer3_Gamma_data;
	*Res_Layer4_Gamma = Res_Layer4_Gamma_data;
	*Res_Layer5_Gamma = Res_Layer5_Gamma_data;
	*Res_Layer6_Gamma = Res_Layer6_Gamma_data;
	*Res_Layer7_Gamma = Res_Layer7_Gamma_data;
	*Res_Layer8_Gamma = Res_Layer8_Gamma_data;
	*Res_Layer9_Gamma = Res_Layer9_Gamma_data;
	*Res_Layer10_Gamma = Res_Layer10_Gamma_data;
	*Res_Layer11_Gamma = Res_Layer11_Gamma_data;
	*Res_Layer12_Gamma = Res_Layer12_Gamma_data;
	*Res_Layer13_Gamma = Res_Layer13_Gamma_data;
	*Res_Layer14_Gamma = Res_Layer14_Gamma_data;
	*Res_Layer15_Gamma = Res_Layer15_Gamma_data;
	*Res_Layer16_Gamma = Res_Layer16_Gamma_data;
	*Res_Layer17_Gamma = Res_Layer17_Gamma_data;
	*Res_Block3_Gamma = Res_Block3_Gamma_data;
	*Res_Block4_Gamma = Res_Block4_Gamma_data;
	*Res_Block5_Gamma = Res_Block5_Gamma_data;

	*Res_Layer1_Beta = Res_Layer1_Beta_data;
	*Res_Layer2_Beta = Res_Layer2_Beta_data;
	*Res_Layer3_Beta = Res_Layer3_Beta_data;
	*Res_Layer4_Beta = Res_Layer4_Beta_data;
	*Res_Layer5_Beta = Res_Layer5_Beta_data;
	*Res_Layer6_Beta = Res_Layer6_Beta_data;
	*Res_Layer7_Beta = Res_Layer7_Beta_data;
	*Res_Layer8_Beta = Res_Layer8_Beta_data;
	*Res_Layer9_Beta = Res_Layer9_Beta_data;
	*Res_Layer10_Beta = Res_Layer10_Beta_data;
	*Res_Layer11_Beta = Res_Layer11_Beta_data;
	*Res_Layer12_Beta = Res_Layer12_Beta_data;
	*Res_Layer13_Beta = Res_Layer13_Beta_data;
	*Res_Layer14_Beta = Res_Layer14_Beta_data;
	*Res_Layer15_Beta = Res_Layer15_Beta_data;
	*Res_Layer16_Beta = Res_Layer16_Beta_data;
	*Res_Layer17_Beta = Res_Layer17_Beta_data;
	*Res_Block3_Beta = Res_Block3_Beta_data;
	*Res_Block4_Beta = Res_Block4_Beta_data;
	*Res_Block5_Beta = Res_Block5_Beta_data;

	*Res_mean1 = Res_mean1_data;
	*Res_mean2 = Res_mean2_data;
	*Res_mean3 = Res_mean3_data;
	*Res_mean4 = Res_mean4_data;
	*Res_mean5 = Res_mean5_data;
	*Res_mean6 = Res_mean6_data;
	*Res_mean7 = Res_mean7_data;
	*Res_mean8 = Res_mean8_data;
	*Res_mean9 = Res_mean9_data;
	*Res_mean10 = Res_mean10_data;
	*Res_mean11 = Res_mean11_data;
	*Res_mean12 = Res_mean12_data;
	*Res_mean13 = Res_mean13_data;
	*Res_mean14 = Res_mean14_data;
	*Res_mean15 = Res_mean15_data;
	*Res_mean16 = Res_mean16_data;
	*Res_mean17 = Res_mean17_data;
	*Res_Block3_mean = Res_Block3_mean_data;
	*Res_Block4_mean = Res_Block4_mean_data;
	*Res_Block5_mean = Res_Block5_mean_data;

	*Res_var1 = Res_var1_data;
	*Res_var2 = Res_var2_data;
	*Res_var3 = Res_var3_data;
	*Res_var4 = Res_var4_data;
	*Res_var5 = Res_var5_data;
	*Res_var6 = Res_var6_data;
	*Res_var7 = Res_var7_data;
	*Res_var8 = Res_var8_data;
	*Res_var9 = Res_var9_data;
	*Res_var10 = Res_var10_data;
	*Res_var11 = Res_var11_data;
	*Res_var12 = Res_var12_data;
	*Res_var13 = Res_var13_data;
	*Res_var14 = Res_var14_data;
	*Res_var15 = Res_var15_data;
	*Res_var16 = Res_var16_data;
	*Res_var17 = Res_var17_data;
	*Res_Block3_var = Res_Block3_var_data;
	*Res_Block4_var = Res_Block4_var_data;
	*Res_Block5_var = Res_Block5_var_data;

	*Res_FC_bias = Res_FC_bias_data;
	*Res_FC_Weights = Res_FC_Weights_data;

	free(Res_Layer1_Neurons_CPU);

	free(Res_Layer1_Weights_CPU);
    free(Res_Layer2_Weights_CPU);
    free(Res_Layer3_Weights_CPU);
    free(Res_Layer4_Weights_CPU);
    free(Res_Layer5_Weights_CPU);
    free(Res_Layer6_Weights_CPU);
    free(Res_Layer7_Weights_CPU);
    free(Res_Layer8_Weights_CPU);
	free(Res_Layer9_Weights_CPU);
    free(Res_Layer10_Weights_CPU);
    free(Res_Layer11_Weights_CPU);
    free(Res_Layer12_Weights_CPU);
    free(Res_Layer13_Weights_CPU);
    free(Res_Layer14_Weights_CPU);
    free(Res_Layer15_Weights_CPU);
    free(Res_Layer16_Weights_CPU);
	free(Res_Layer17_Weights_CPU);
    free(Res_Block3_Weights_CPU);
    free(Res_Block4_Weights_CPU);
    free(Res_Block5_Weights_CPU);

	free(Res_Layer1_Gamma_CPU);
    free(Res_Layer2_Gamma_CPU);
    free(Res_Layer3_Gamma_CPU);
    free(Res_Layer4_Gamma_CPU);
    free(Res_Layer5_Gamma_CPU);
    free(Res_Layer6_Gamma_CPU);
    free(Res_Layer7_Gamma_CPU);
    free(Res_Layer8_Gamma_CPU);
	free(Res_Layer9_Gamma_CPU);
    free(Res_Layer10_Gamma_CPU);
    free(Res_Layer11_Gamma_CPU);
    free(Res_Layer12_Gamma_CPU);
    free(Res_Layer13_Gamma_CPU);
    free(Res_Layer14_Gamma_CPU);
    free(Res_Layer15_Gamma_CPU);
    free(Res_Layer16_Gamma_CPU);
	free(Res_Layer17_Gamma_CPU);
    free(Res_Block3_Gamma_CPU);
    free(Res_Block4_Gamma_CPU);
    free(Res_Block5_Gamma_CPU);

	free(Res_Layer1_Beta_CPU);
    free(Res_Layer2_Beta_CPU);
    free(Res_Layer3_Beta_CPU);
    free(Res_Layer4_Beta_CPU);
    free(Res_Layer5_Beta_CPU);
    free(Res_Layer6_Beta_CPU);
    free(Res_Layer7_Beta_CPU);
    free(Res_Layer8_Beta_CPU);
	free(Res_Layer9_Beta_CPU);
    free(Res_Layer10_Beta_CPU);
    free(Res_Layer11_Beta_CPU);
    free(Res_Layer12_Beta_CPU);
    free(Res_Layer13_Beta_CPU);
    free(Res_Layer14_Beta_CPU);
    free(Res_Layer15_Beta_CPU);
    free(Res_Layer16_Beta_CPU);
	free(Res_Layer17_Beta_CPU);
    free(Res_Block3_Beta_CPU);
    free(Res_Block4_Beta_CPU);
    free(Res_Block5_Beta_CPU);

	free(Res_mean1_CPU);
	free(Res_mean2_CPU);
	free(Res_mean3_CPU);
	free(Res_mean4_CPU);
	free(Res_mean5_CPU);
	free(Res_mean6_CPU);
	free(Res_mean7_CPU);
	free(Res_mean8_CPU);
	free(Res_mean9_CPU);
	free(Res_mean10_CPU);
	free(Res_mean11_CPU);
	free(Res_mean12_CPU);
	free(Res_mean13_CPU);
	free(Res_mean14_CPU);
	free(Res_mean15_CPU);
	free(Res_mean16_CPU);
	free(Res_mean17_CPU);
	free(Res_Block3_mean_CPU);
	free(Res_Block4_mean_CPU);
	free(Res_Block5_mean_CPU);

	free(Res_var1_CPU);
	free(Res_var2_CPU);
	free(Res_var3_CPU);
	free(Res_var4_CPU);
	free(Res_var5_CPU);
	free(Res_var6_CPU);
	free(Res_var7_CPU);
	free(Res_var8_CPU);
	free(Res_var9_CPU);
	free(Res_var10_CPU);
	free(Res_var11_CPU);
	free(Res_var12_CPU);
	free(Res_var13_CPU);
	free(Res_var14_CPU);
	free(Res_var15_CPU);
	free(Res_var16_CPU);
	free(Res_var17_CPU);
	free(Res_Block3_var_CPU);
	free(Res_Block4_var_CPU);
	free(Res_Block5_var_CPU);

	float *Res_Layer1_bn_data; 
	cudaMalloc((void**) &Res_Layer1_bn_data, (64*112*112) * sizeof(float)); //64*112*112
	*Res_Layer1_bn = Res_Layer1_bn_data;

	float *Res_Layer1_pool_data;
    cudaMalloc((void**) &Res_Layer1_pool_data, (64*112*112) * sizeof(float)); //64*112*112
	*Res_Layer1_pool = Res_Layer1_pool_data;

    float *Res_Layer2_Neurons_data;
    cudaMalloc((void**) &Res_Layer2_Neurons_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer2_Neurons = Res_Layer2_Neurons_data;

    float *Res_Layer2_bn_data;
    cudaMalloc((void**) &Res_Layer2_bn_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer2_bn = Res_Layer2_bn_data;

    float *Res_Layer3_Neurons_data;
	cudaMalloc((void**) &Res_Layer3_Neurons_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer3_Neurons = Res_Layer3_Neurons_data;

    float *Res_Layer3_bn_data;
	cudaMalloc((void**) &Res_Layer3_bn_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer3_bn = Res_Layer3_bn_data;

    float *Res_Layer3_basic_data;
    cudaMalloc((void**) &Res_Layer3_basic_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer3_basic = Res_Layer3_basic_data;

    float *Res_Layer4_Neurons_data;
    cudaMalloc((void**) &Res_Layer4_Neurons_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer4_Neurons = Res_Layer4_Neurons_data;

    float *Res_Layer4_bn_data;
    cudaMalloc((void**) &Res_Layer4_bn_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer4_bn = Res_Layer4_bn_data;
	
    float *Res_Layer5_Neurons_data;
    cudaMalloc((void**) &Res_Layer5_Neurons_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer5_Neurons = Res_Layer5_Neurons_data;

    float *Res_Layer5_bn_data;
    cudaMalloc((void**) &Res_Layer5_bn_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer5_bn = Res_Layer5_bn_data;

    float *Res_Layer5_basic_data;
    cudaMalloc((void**) &Res_Layer5_basic_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer5_basic = Res_Layer5_basic_data;

    float *Res_Layer6_Neurons_data;
    cudaMalloc((void**) &Res_Layer6_Neurons_data, (64*56*56) * sizeof(float)); //64*56*56
	*Res_Layer6_Neurons = Res_Layer6_Neurons_data;

    float *Res_Layer6_bn_data;
    cudaMalloc((void**) &Res_Layer6_bn_data, sizeof(float) * (128*28*28)); //128*28*28
	*Res_Layer6_bn = Res_Layer6_bn_data;

    float *Res_Layer7_Neurons_data;
    cudaMalloc((void**) &Res_Layer7_Neurons_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer7_Neurons = Res_Layer7_Neurons_data;

    float *Res_Layer7_bn_data;
    cudaMalloc((void**) &Res_Layer7_bn_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer7_bn = Res_Layer7_bn_data;

    float *Res_Layer7_basic_data;
    cudaMalloc((void**) &Res_Layer7_basic_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer7_basic = Res_Layer7_basic_data;

    float *Res_Block3_bn_data, *Res_Block3_basic_data, *Res_Layer8_Neurons_data;
	cudaMalloc((void**) &Res_Block3_bn_data, (128*28*28) * sizeof(float)); //128*28*28
	cudaMalloc((void**) &Res_Block3_basic_data, (128*28*28) * sizeof(float)); //128*28*28
	cudaMalloc((void**) &Res_Layer8_Neurons_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Block3_bn = Res_Block3_bn_data;
	*Res_Block3_basic = Res_Block3_basic_data;
	*Res_Layer8_Neurons = Res_Layer8_Neurons_data;

    float *Res_Layer8_bn_data;
    cudaMalloc((void**) &Res_Layer8_bn_data,(128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer8_bn = Res_Layer8_bn_data;

    float *Res_Layer9_Neurons_data;
    cudaMalloc((void**) &Res_Layer9_Neurons_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer9_Neurons = Res_Layer9_Neurons_data;

    float *Res_Layer9_bn_data;
    cudaMalloc((void**) &Res_Layer9_bn_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer9_bn = Res_Layer9_bn_data;

    float *Res_Layer9_basic_data;
    cudaMalloc((void**) &Res_Layer9_basic_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer9_basic = Res_Layer9_basic_data;

    float *Res_Layer10_Neurons_data;
	cudaMalloc((void**) &Res_Layer10_Neurons_data, (128*28*28) * sizeof(float)); //128*28*28
	*Res_Layer10_Neurons = Res_Layer10_Neurons_data;

    float *Res_Layer10_bn_data;
    cudaMalloc((void**) &Res_Layer10_bn_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer10_bn = Res_Layer10_bn_data;

    float *Res_Layer11_Neurons_data;
    cudaMalloc((void**) &Res_Layer11_Neurons_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer11_Neurons = Res_Layer11_Neurons_data;

    float *Res_Layer11_bn_data;
    cudaMalloc((void**) &Res_Layer11_bn_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer11_bn = Res_Layer11_bn_data;

    float *Res_Layer11_basic_data;
    cudaMalloc((void**) &Res_Layer11_basic_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer11_basic = Res_Layer11_basic_data;

	float *Res_Block4_bn_data, *Res_Block4_basic_data, *Res_Layer12_Neurons_data;
	cudaMalloc((void**) &Res_Block4_bn_data, (256*14*14) * sizeof(float)); //256*14*14
	cudaMalloc((void**) &Res_Block4_basic_data, (256*14*14) * sizeof(float)); //256*14*14
	cudaMalloc((void**) &Res_Layer12_Neurons_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Block4_bn = Res_Block4_bn_data;
	*Res_Block4_basic = Res_Block4_basic_data;
	*Res_Layer12_Neurons = Res_Layer12_Neurons_data;

    float *Res_Layer12_bn_data;
    cudaMalloc((void**) &Res_Layer12_bn_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer12_bn = Res_Layer12_bn_data;

    float *Res_Layer13_Neurons_data;
    cudaMalloc((void**) &Res_Layer13_Neurons_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer13_Neurons = Res_Layer13_Neurons_data;

    float *Res_Layer13_bn_data;
    cudaMalloc((void**) &Res_Layer13_bn_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer13_bn = Res_Layer13_bn_data;

    float *Res_Layer13_basic_data;
    cudaMalloc((void**) &Res_Layer13_basic_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer13_basic = Res_Layer13_basic_data;

    float *Res_Layer14_Neurons_data;
    cudaMalloc((void**) &Res_Layer14_Neurons_data, (256*14*14) * sizeof(float)); //256*14*14
	*Res_Layer14_Neurons = Res_Layer14_Neurons_data;

    float *Res_Layer14_bn_data, *Res_Layer15_Neurons_data;
	cudaMalloc((void**) &Res_Layer14_bn_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Layer15_Neurons_data, (512*7*7) * sizeof(float)); //512*7*7
	*Res_Layer14_bn = Res_Layer14_bn_data;
	*Res_Layer15_Neurons = Res_Layer15_Neurons_data;

    float *Res_Layer15_bn_data, *Res_Layer15_basic_data;
	cudaMalloc((void**) &Res_Layer15_bn_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Layer15_basic_data, (512*7*7) * sizeof(float)); //512*7*7
	*Res_Layer15_bn = Res_Layer15_bn_data;
	*Res_Layer15_basic = Res_Layer15_basic_data;

	float *Res_Block5_bn_data, *Res_Block5_basic_data, *Res_Layer16_Neurons_data;
	cudaMalloc((void**) &Res_Block5_bn_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Block5_basic_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Layer16_Neurons_data, (512*7*7) * sizeof(float)); //512*7*7
	*Res_Block5_bn = Res_Block5_bn_data;
	*Res_Block5_basic = Res_Block5_basic_data;
	*Res_Layer16_Neurons = Res_Layer16_Neurons_data;

	float *Res_Layer16_bn_data, *Res_Layer17_Neurons_data;
	cudaMalloc((void**) &Res_Layer16_bn_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Layer17_Neurons_data, (512*7*7) * sizeof(float)); //512*7*7
	*Res_Layer16_bn = Res_Layer16_bn_data;
	*Res_Layer17_Neurons = Res_Layer17_Neurons_data;

    float *Res_Layer17_bn_data, *Res_Layer17_basic_data, *Res_Layer18_Neurons_data;
	cudaMalloc((void**) &Res_Layer17_bn_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Layer17_basic_data, (512*7*7) * sizeof(float)); //512*7*7
	cudaMalloc((void**) &Res_Layer18_Neurons_data, (512*7*7) * sizeof(float)); //512*7*7
	*Res_Layer17_bn = Res_Layer17_bn_data;
	*Res_Layer17_basic = Res_Layer17_basic_data;
	*Res_Layer18_Neurons = Res_Layer18_Neurons_data;

    float *Res_FC_Neurons_data;
	cudaMalloc((void**) &Res_FC_Neurons_data, 512 * sizeof(float));
	*Res_FC_Neurons = Res_FC_Neurons_data;

    float *Res_Result_Neurons_data;
    cudaMalloc((void**) &Res_Result_Neurons_data, 1000 * sizeof(float)); //1000
	*Res_Result_Neurons = Res_Result_Neurons_data;
}

void res_first_conv(float *Res_Layer1_Neurons,float *Res_Layer1_Weights,float *Res_Layer1_bn,float *Res_Layer1_pool,float *Res_mean1,float *Res_var1,float *Res_Layer1_Gamma,float *Res_Layer1_Beta)
{
    dim3 Block1_Block(64,16,16);
    dim3 Block_Thread(7,7);
	first<<<Block1_Block,Block_Thread>>>(NULL,Res_Layer1_Neurons,Res_Layer1_Weights,Res_Layer1_bn,224,112,2,3,7,3,false,true);
	batchnorm<<<Block1_Block,Block_Thread>>>(Res_Layer1_bn,Res_Layer1_pool,Res_mean1,Res_var1,Res_Layer1_Gamma,Res_Layer1_Beta,112,true);
}

void res_first_pool(float *Res_Layer1_pool,float *Res_Layer2_Neurons)
{
    dim3 Block1_Pool_Block(64,8,8);
    dim3 Block_Thread(7,7);
	max<<<Block1_Pool_Block,Block_Thread>>>(Res_Layer1_pool,Res_Layer2_Neurons,112,56,2,1,3);
}

void res_second_conv(float *Res_Layer2_Neurons,float *Res_Layer2_Weights,float *Res_Layer2_bn,float *Res_Layer3_Neurons,float *Res_mean2,float *Res_var2,float *Res_Layer2_Gamma,float *Res_Layer2_Beta)
{
    dim3 Block2_Block(64,8,8);
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block2_Block,Block_Thread>>>(NULL,Res_Layer2_Neurons,Res_Layer2_Weights,Res_Layer2_bn,56,56,1,1,3,64,false,false);
	batchnorm<<<Block2_Block,Block_Thread>>>(Res_Layer2_bn,Res_Layer3_Neurons,Res_mean2,Res_var2,Res_Layer2_Gamma,Res_Layer2_Beta,56,true);
}

void res_third_conv(float *Res_Layer3_Neurons,float *Res_Layer3_Weights,float *Res_Layer3_bn,float *Res_Layer3_basic,float *Res_mean3,float *Res_var3,float *Res_Layer3_Gamma,float *Res_Layer3_Beta)
{
    dim3 Block2_Block(64,8,8);
    dim3 Block_Thread(7,7);
 	ResNet_conv<<<Block2_Block,Block_Thread>>>(NULL,Res_Layer3_Neurons,Res_Layer3_Weights,Res_Layer3_bn,56,56,1,1,3,64,false,false);
	batchnorm<<<Block2_Block,Block_Thread>>>(Res_Layer3_bn,Res_Layer3_basic,Res_mean3,Res_var3,Res_Layer3_Gamma,Res_Layer3_Beta,56,false);   
}

void res_third_basic(float *Res_Layer2_Neurons,float *Res_Layer3_basic,float *Res_Layer4_Neurons)
{
    dim3 Block2_Block(64,8,8);
    dim3 Block_Thread(7,7);
	basic_block<<<Block2_Block,Block_Thread>>>(Res_Layer2_Neurons,Res_Layer3_basic,Res_Layer4_Neurons,56,true);   
}

void res_fourth_conv(float *Res_Layer4_Neurons,float *Res_Layer4_Weights,float *Res_Layer4_bn,float *Res_Layer5_Neurons,float *Res_mean4,float *Res_var4,float *Res_Layer4_Gamma,float *Res_Layer4_Beta)
{
    dim3 Block2_Block(64,8,8);
    dim3 Block_Thread(7,7);
  	ResNet_conv<<<Block2_Block,Block_Thread>>>(NULL,Res_Layer4_Neurons,Res_Layer4_Weights,Res_Layer4_bn,56,56,1,1,3,64,false,false);
	batchnorm<<<Block2_Block,Block_Thread>>>(Res_Layer4_bn,Res_Layer5_Neurons,Res_mean4,Res_var4,Res_Layer4_Gamma,Res_Layer4_Beta,56,true);
}

void res_fifth_conv(float *Res_Layer5_Neurons,float *Res_Layer5_Weights,float *Res_Layer5_bn,float *Res_Layer5_basic,float *Res_mean5,float *Res_var5,float *Res_Layer5_Gamma,float *Res_Layer5_Beta)
{
    dim3 Block2_Block(64,8,8);
    dim3 Block_Thread(7,7);    
	ResNet_conv<<<Block2_Block,Block_Thread>>>(NULL,Res_Layer5_Neurons,Res_Layer5_Weights,Res_Layer5_bn,56,56,1,1,3,64,false,false);
	batchnorm<<<Block2_Block,Block_Thread>>>(Res_Layer5_bn,Res_Layer5_basic,Res_mean5,Res_var5,Res_Layer5_Gamma,Res_Layer5_Beta,56,false);
}

void res_fifth_basic(float *Res_Layer4_Neurons,float *Res_Layer5_basic,float *Res_Layer6_Neurons)
{
    dim3 Block2_Block(64,8,8);
    dim3 Block_Thread(7,7); 
	basic_block<<<Block2_Block,Block_Thread>>>(Res_Layer4_Neurons,Res_Layer5_basic,Res_Layer6_Neurons,56,true);
}

void res_sixth_conv(float *Res_Layer6_Neurons,float *Res_Layer6_Weights,float *Res_Layer6_bn,float *Res_Layer7_Neurons,float *Res_mean6,float *Res_var6,float *Res_Layer6_Gamma,float *Res_Layer6_Beta)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
	ResNet_conv<<<Block3_Block,Block_Thread>>>(NULL,Res_Layer6_Neurons,Res_Layer6_Weights,Res_Layer6_bn,56,28,2,1,3,64,false,false);
	batchnorm<<<Block3_Block,Block_Thread>>>(Res_Layer6_bn,Res_Layer7_Neurons,Res_mean6,Res_var6,Res_Layer6_Gamma,Res_Layer6_Beta,28,true);
}

void res_seventh_conv(float *Res_Layer7_Neurons,float *Res_Layer7_Weights,float *Res_Layer7_bn,float *Res_Layer7_basic,float *Res_mean7,float *Res_var7,float *Res_Layer7_Gamma,float *Res_Layer7_Beta)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
	ResNet_conv<<<Block3_Block,Block_Thread>>>(NULL,Res_Layer7_Neurons,Res_Layer7_Weights,Res_Layer7_bn,28,28,1,1,3,128,false,false);
	batchnorm<<<Block3_Block,Block_Thread>>>(Res_Layer7_bn,Res_Layer7_basic,Res_mean7,Res_var7,Res_Layer7_Gamma,Res_Layer7_Beta,28,false);
}

void res_Block_B_conv(float *Res_Layer6_Neurons,float *Res_Block3_Weights,float *Res_Block3_bn,float *Res_Block3_basic,float *Res_Block3_mean,float *Res_Block3_var,float *Res_Block3_Gamma,float *Res_Block3_Beta)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
	ResNet_conv<<<Block3_Block,Block_Thread>>>(NULL,Res_Layer6_Neurons,Res_Block3_Weights,Res_Block3_bn,56,28,2,0,1,64,false,false); 
	batchnorm<<<Block3_Block,Block_Thread>>>(Res_Block3_bn,Res_Block3_basic,Res_Block3_mean,Res_Block3_var,Res_Block3_Gamma,Res_Block3_Beta,28,false);
}

void res_Block_B_basic(float *Res_Layer7_basic,float *Res_Block3_basic,float *Res_Layer8_Neurons)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
	basic_block<<<Block3_Block,Block_Thread>>>(Res_Layer7_basic,Res_Block3_basic,Res_Layer8_Neurons,28,true);
}

void res_eighth_conv(float *Res_Layer8_Neurons,float *Res_Layer8_Weights,float *Res_Layer8_bn,float *Res_Layer9_Neurons,float *Res_mean8,float *Res_var8,float *Res_Layer8_Gamma,float *Res_Layer8_Beta)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
	ResNet_conv<<<Block3_Block,Block_Thread>>>(NULL,Res_Layer8_Neurons,Res_Layer8_Weights,Res_Layer8_bn,28,28,1,1,3,128,false,false);
	batchnorm<<<Block3_Block,Block_Thread>>>(Res_Layer8_bn,Res_Layer9_Neurons,Res_mean8,Res_var8,Res_Layer8_Gamma,Res_Layer8_Beta,28,true);
}

void res_ninth_conv(float *Res_Layer9_Neurons,float *Res_Layer9_Weights,float *Res_Layer9_bn,float *Res_Layer9_basic,float *Res_mean9,float *Res_var9,float *Res_Layer9_Gamma,float *Res_Layer9_Beta)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
	ResNet_conv<<<Block3_Block,Block_Thread>>>(NULL,Res_Layer9_Neurons,Res_Layer9_Weights,Res_Layer9_bn,28,28,1,1,3,128,false,false);
	batchnorm<<<Block3_Block,Block_Thread>>>(Res_Layer9_bn,Res_Layer9_basic,Res_mean9,Res_var9,Res_Layer9_Gamma,Res_Layer9_Beta,28,false);
}

void res_ninth_basic(float *Res_Layer8_Neurons,float *Res_Layer9_basic,float *Res_Layer10_Neurons)
{
    dim3 Block3_Block(128,4,4);
    dim3 Block_Thread(7,7); 
 	basic_block<<<Block3_Block,Block_Thread>>>(Res_Layer8_Neurons,Res_Layer9_basic,Res_Layer10_Neurons,28,true);   
}

void res_tenth_conv(float *Res_Layer10_Neurons,float *Res_Layer10_Weights,float *Res_Layer10_bn,float *Res_Layer11_Neurons,float *Res_mean10,float *Res_var10,float *Res_Layer10_Gamma,float *Res_Layer10_Beta)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7); 
	ResNet_conv<<<Block4_Block,Block_Thread>>>(NULL,Res_Layer10_Neurons,Res_Layer10_Weights,Res_Layer10_bn,28,14,2,1,3,128,false,false);
	batchnorm<<<Block4_Block,Block_Thread>>>(Res_Layer10_bn,Res_Layer11_Neurons,Res_mean10,Res_var10,Res_Layer10_Gamma,Res_Layer10_Beta,14,true);
}

void res_eleventh_conv(float *Res_Layer11_Neurons,float *Res_Layer11_Weights,float *Res_Layer11_bn,float *Res_Layer11_basic,float *Res_mean11,float *Res_var11,float *Res_Layer11_Gamma,float *Res_Layer11_Beta)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block4_Block,Block_Thread>>>(NULL,Res_Layer11_Neurons,Res_Layer11_Weights,Res_Layer11_bn,14,14,1,1,3,256,false,false);
	batchnorm<<<Block4_Block,Block_Thread>>>(Res_Layer11_bn,Res_Layer11_basic,Res_mean11,Res_var11,Res_Layer11_Gamma,Res_Layer11_Beta,14,false);
}

void res_Block_C_conv(float *Res_Layer10_Neurons,float *Res_Block4_Weights,float *Res_Block4_bn,float *Res_Block4_basic,float *Res_Block4_mean,float *Res_Block4_var,float *Res_Block4_Gamma,float *Res_Block4_Beta)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block4_Block,Block_Thread>>>(NULL,Res_Layer10_Neurons,Res_Block4_Weights,Res_Block4_bn,28,14,2,0,1,128,false,false);
	batchnorm<<<Block4_Block,Block_Thread>>>(Res_Block4_bn,Res_Block4_basic,Res_Block4_mean,Res_Block4_var,Res_Block4_Gamma,Res_Block4_Beta,14,false);
}

void res_Block_C_basic(float *Res_Layer11_basic,float *Res_Block4_basic,float *Res_Layer12_Neurons)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7);
	basic_block<<<Block4_Block,Block_Thread>>>(Res_Layer11_basic,Res_Block4_basic,Res_Layer12_Neurons,14,true);
}

void res_twelfth_conv(float *Res_Layer12_Neurons,float *Res_Layer12_Weights,float *Res_Layer12_bn,float *Res_Layer13_Neurons,float *Res_mean12,float *Res_var12,float *Res_Layer12_Gamma,float *Res_Layer12_Beta)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block4_Block,Block_Thread>>>(NULL,Res_Layer12_Neurons,Res_Layer12_Weights,Res_Layer12_bn,14,14,1,1,3,256,false,false);
	batchnorm<<<Block4_Block,Block_Thread>>>(Res_Layer12_bn,Res_Layer13_Neurons,Res_mean12,Res_var12,Res_Layer12_Gamma,Res_Layer12_Beta,14,true);
}

void res_thirteenth_conv(float *Res_Layer13_Neurons,float *Res_Layer13_Weights,float *Res_Layer13_bn,float *Res_Layer13_basic,float *Res_mean13,float *Res_var13,float *Res_Layer13_Gamma,float *Res_Layer13_Beta)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block4_Block,Block_Thread>>>(NULL,Res_Layer13_Neurons,Res_Layer13_Weights,Res_Layer13_bn,14,14,1,1,3,256,false,false); 
	batchnorm<<<Block4_Block,Block_Thread>>>(Res_Layer13_bn,Res_Layer13_basic,Res_mean13,Res_var13,Res_Layer13_Gamma,Res_Layer13_Beta,14,false);
}

void res_thirteenth_basic(float *Res_Layer12_Neurons,float *Res_Layer13_basic,float *Res_Layer14_Neurons)
{
    dim3 Block4_Block(256,2,2);
    dim3 Block_Thread(7,7);
	basic_block<<<Block4_Block,Block_Thread>>>(Res_Layer12_Neurons,Res_Layer13_basic,Res_Layer14_Neurons,14,true);
}

void res_fourteenth_conv(float *Res_Layer14_Neurons,float *Res_Layer14_Weights,float *Res_Layer14_bn,float *Res_Layer15_Neurons,float *Res_mean14,float *Res_var14,float *Res_Layer14_Gamma,float *Res_Layer14_Beta)
{
	dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
    ResNet_conv<<<Block5_Block,Block_Thread>>>(NULL,Res_Layer14_Neurons,Res_Layer14_Weights,Res_Layer14_bn,14,7,2,1,3,256,false,false);
	batchnorm<<<Block5_Block,Block_Thread>>>(Res_Layer14_bn,Res_Layer15_Neurons,Res_mean14,Res_var14,Res_Layer14_Gamma,Res_Layer14_Beta,7,true);
}

void res_fifteenth_conv(float *Res_Layer15_Neurons,float *Res_Layer15_Weights,float *Res_Layer15_bn,float *Res_Layer15_basic,float *Res_mean15,float *Res_var15,float *Res_Layer15_Gamma,float *Res_Layer15_Beta)
{
    dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block5_Block,Block_Thread>>>(NULL,Res_Layer15_Neurons,Res_Layer15_Weights,Res_Layer15_bn,7,7,1,1,3,512,false,false);
	batchnorm<<<Block5_Block,Block_Thread>>>(Res_Layer15_bn,Res_Layer15_basic,Res_mean15,Res_var15,Res_Layer15_Gamma,Res_Layer15_Beta,7,false);
}

void res_Block_D_conv(float *Res_Layer14_Neurons,float *Res_Block5_Weights,float *Res_Block5_bn,float *Res_Block5_basic,float *Res_Block5_mean,float *Res_Block5_var,float *Res_Block5_Gamma,float *Res_Block5_Beta)
{
	dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block5_Block,Block_Thread>>>(NULL,Res_Layer14_Neurons,Res_Block5_Weights,Res_Block5_bn,14,7,2,0,1,256,false,false);
	batchnorm<<<Block5_Block,Block_Thread>>>(Res_Block5_bn,Res_Block5_basic,Res_Block5_mean,Res_Block5_var,Res_Block5_Gamma,Res_Block5_Beta,7,false);
}

void res_Block_D_basic(float *Res_Layer15_basic,float *Res_Block5_basic,float *Res_Layer16_Neurons)
{
	dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
	basic_block<<<Block5_Block,Block_Thread>>>(Res_Layer15_basic,Res_Block5_basic,Res_Layer16_Neurons,7,true);
}

void res_sixteenth_conv(float *Res_Layer16_Neurons,float *Res_Layer16_Weights,float *Res_Layer16_bn,float *Res_Layer17_Neurons,float *Res_mean16,float *Res_var16,float *Res_Layer16_Gamma,float *Res_Layer16_Beta)
{
	dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block5_Block,Block_Thread>>>(NULL,Res_Layer16_Neurons,Res_Layer16_Weights,Res_Layer16_bn,7,7,1,1,3,512,false,false);
	batchnorm<<<Block5_Block,Block_Thread>>>(Res_Layer16_bn,Res_Layer17_Neurons,Res_mean16,Res_var16,Res_Layer16_Gamma,Res_Layer16_Beta,7,true);
}

void res_seventeenth_conv(float *Res_Layer17_Neurons,float *Res_Layer17_Weights,float *Res_Layer17_bn,float *Res_Layer17_basic,float *Res_mean17,float *Res_var17,float *Res_Layer17_Gamma,float *Res_Layer17_Beta)
{
	dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
	ResNet_conv<<<Block5_Block,Block_Thread>>>(NULL,Res_Layer17_Neurons,Res_Layer17_Weights,Res_Layer17_bn,7,7,1,1,3,512,false,false); 
	batchnorm<<<Block5_Block,Block_Thread>>>(Res_Layer17_bn,Res_Layer17_basic,Res_mean17,Res_var17,Res_Layer17_Gamma,Res_Layer17_Beta,7,false);
}

void res_seventeenth_basic(float *Res_Layer16_Neurons,float *Res_Layer17_basic,float *Res_Layer18_Neurons)
{
	dim3 Block5_Block(512,1,1);    
    dim3 Block_Thread(7,7);
	basic_block<<<Block5_Block,Block_Thread>>>(Res_Layer16_Neurons,Res_Layer17_basic,Res_Layer18_Neurons,7,true);
}

void res_avg_pool(float *Res_Layer18_Neurons,float *Res_FC_Neurons)
{
    dim3 Avg_Block(512,1,1);
    dim3 Single_Thread(1,1);
	globalavg<<<Avg_Block,Single_Thread>>>(Res_Layer18_Neurons,Res_FC_Neurons,7);
}

void res_fc(float *Res_FC_bias,float *Res_FC_Neurons,float *Res_FC_Weights,float *Res_Result_Neurons)
{
	dim3 FC_Block(1000,1,1);
    dim3 Single_Thread(1,1);
	ResNet_fc<<<FC_Block,Single_Thread>>>(Res_FC_bias,Res_FC_Neurons,Res_FC_Weights,Res_Result_Neurons,512,false);

    float *Res_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
	cudaMemcpy(Res_Result_Neurons_CPU, Res_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

	float max1 = 0.0;
	int index1 = 0; 
	for(int i = 0; i < 1000; i++){
		if(max1 < Res_Result_Neurons_CPU[i]){
			max1 = Res_Result_Neurons_CPU[i];	
			index1 = i;
		}
	}
	
	int line_count1 = 0;
	char buffer[1000];
	FILE *list1 = fopen("common/imagenet1000_clsidx_to_labels.txt","rt");
	while(fgets(buffer, 1000, list1) != NULL){
		line_count1++;
		if(line_count1 == (index1+1)){
			// printf("\n---Resnet18 Result---");
			// printf("\nClass ID: %d\nClass Name: %sProbability: %f\n", index1, buffer, max1);
			printf("\nResnet18: %d, %s", index1, buffer);
			break;
		}
	}
	fclose(list1);

	free(Res_Result_Neurons_CPU);	
}

void free_resnet18(float *Res_Layer1_Neurons,float *Res_Layer2_Neurons,float *Res_Layer3_Neurons,float *Res_Layer4_Neurons,
					float *Res_Layer5_Neurons,float *Res_Layer6_Neurons,float *Res_Layer7_Neurons,float *Res_Layer8_Neurons,
					float *Res_Layer9_Neurons,float *Res_Layer10_Neurons,float *Res_Layer11_Neurons,float *Res_Layer12_Neurons,
					float *Res_Layer13_Neurons,float *Res_Layer14_Neurons,float *Res_Layer15_Neurons,float *Res_Layer16_Neurons,
					float *Res_Layer17_Neurons,float *Res_Layer18_Neurons,
                    float *Res_Layer1_Weights,float *Res_Layer2_Weights,float *Res_Layer3_Weights,float *Res_Layer4_Weights,
                    float *Res_Layer5_Weights,float *Res_Layer6_Weights,float *Res_Layer7_Weights,float *Res_Layer8_Weights,
                    float *Res_Layer9_Weights,float *Res_Layer10_Weights,float *Res_Layer11_Weights,float *Res_Layer12_Weights,
                    float *Res_Layer13_Weights,float *Res_Layer14_Weights,float *Res_Layer15_Weights,float *Res_Layer16_Weights,
                    float *Res_Layer17_Weights,float *Res_Block3_Weights,float *Res_Block4_Weights,float *Res_Block5_Weights,
                    float *Res_Layer1_Gamma,float *Res_Layer2_Gamma,float *Res_Layer3_Gamma,float *Res_Layer4_Gamma,
                    float *Res_Layer5_Gamma,float *Res_Layer6_Gamma,float *Res_Layer7_Gamma,float *Res_Layer8_Gamma,
                    float *Res_Layer9_Gamma,float *Res_Layer10_Gamma,float *Res_Layer11_Gamma,float *Res_Layer12_Gamma,
                    float *Res_Layer13_Gamma,float *Res_Layer14_Gamma,float *Res_Layer15_Gamma,float *Res_Layer16_Gamma,
                    float *Res_Layer17_Gamma,float *Res_Block3_Gamma,float *Res_Block4_Gamma,float *Res_Block5_Gamma,
                    float *Res_Layer1_Beta,float *Res_Layer2_Beta,float *Res_Layer3_Beta,float *Res_Layer4_Beta,
                    float *Res_Layer5_Beta,float *Res_Layer6_Beta,float *Res_Layer7_Beta,float *Res_Layer8_Beta,
                    float *Res_Layer9_Beta,float *Res_Layer10_Beta,float *Res_Layer11_Beta,float *Res_Layer12_Beta,
                    float *Res_Layer13_Beta,float *Res_Layer14_Beta,float *Res_Layer15_Beta,float *Res_Layer16_Beta,
                    float *Res_Layer17_Beta,float *Res_Block3_Beta,float *Res_Block4_Beta,float *Res_Block5_Beta,
                    float *Res_mean1,float *Res_mean2,float *Res_mean3,float *Res_mean4,float *Res_mean5,
                    float *Res_mean6,float *Res_mean7,float *Res_mean8,float *Res_mean9,float *Res_mean10,
                    float *Res_mean11,float *Res_mean12,float *Res_mean13,float *Res_mean14,float *Res_mean15,
                    float *Res_mean16,float *Res_mean17,float *Res_Block3_mean,float *Res_Block4_mean,float *Res_Block5_mean,
                    float *Res_var1,float *Res_var2,float *Res_var3,float *Res_var4,float *Res_var5,
                    float *Res_var6,float *Res_var7,float *Res_var8,float *Res_var9,float *Res_var10,
                    float *Res_var11,float *Res_var12,float *Res_var13,float *Res_var14,float *Res_var15,
                    float *Res_var16,float *Res_var17,float *Res_Block3_var,float *Res_Block4_var,float *Res_Block5_var,
                    float *Res_FC_bias,float *Res_FC_Weights,
					float *Res_Layer3_basic,float *Res_Layer5_basic,float *Res_Layer7_basic,float *Res_Layer9_basic,
					float *Res_Layer11_basic,float *Res_Layer13_basic,float *Res_Layer15_basic,float *Res_Layer17_basic,
					float *Res_Block3_basic,float *Res_Block4_basic,float *Res_Block5_basic,
					float *Res_Layer1_bn,float *Res_Layer2_bn,float *Res_Layer3_bn,float *Res_Layer4_bn,
					float *Res_Layer5_bn,float *Res_Layer6_bn,float *Res_Layer7_bn,float *Res_Layer8_bn,
					float *Res_Layer9_bn,float *Res_Layer10_bn,float *Res_Layer11_bn,float *Res_Layer12_bn,
					float *Res_Layer13_bn,float *Res_Layer14_bn,float *Res_Layer15_bn,float *Res_Layer16_bn,
					float *Res_Layer17_bn,float *Res_Block3_bn,float *Res_Block4_bn,float *Res_Block5_bn,
					float *Res_Layer1_pool,float *Res_FC_Neurons,float *Res_Result_Neurons)
{
	cudaFree(Res_Layer1_Neurons);
    cudaFree(Res_Layer2_Neurons);
	cudaFree(Res_Layer3_Neurons);
	cudaFree(Res_Layer4_Neurons);
	cudaFree(Res_Layer5_Neurons);
	cudaFree(Res_Layer6_Neurons);
	cudaFree(Res_Layer7_Neurons);
	cudaFree(Res_Layer8_Neurons);
	cudaFree(Res_Layer9_Neurons);
	cudaFree(Res_Layer10_Neurons);
	cudaFree(Res_Layer11_Neurons);
	cudaFree(Res_Layer12_Neurons);
	cudaFree(Res_Layer13_Neurons);
	cudaFree(Res_Layer14_Neurons);
	cudaFree(Res_Layer15_Neurons);
	cudaFree(Res_Layer16_Neurons);
	cudaFree(Res_Layer17_Neurons);
	cudaFree(Res_Layer18_Neurons);

	cudaFree(Res_Layer1_Weights);
	cudaFree(Res_Layer2_Weights);
	cudaFree(Res_Layer3_Weights);
	cudaFree(Res_Layer4_Weights);
	cudaFree(Res_Layer5_Weights);
	cudaFree(Res_Layer6_Weights);
	cudaFree(Res_Layer7_Weights);
	cudaFree(Res_Layer8_Weights);
	cudaFree(Res_Layer9_Weights);
	cudaFree(Res_Layer10_Weights);
	cudaFree(Res_Layer11_Weights);
	cudaFree(Res_Layer12_Weights);
	cudaFree(Res_Layer13_Weights);
	cudaFree(Res_Layer14_Weights);
	cudaFree(Res_Layer15_Weights);
	cudaFree(Res_Layer16_Weights);
	cudaFree(Res_Layer17_Weights);
	cudaFree(Res_Block3_Weights);
	cudaFree(Res_Block4_Weights);
	cudaFree(Res_Block5_Weights);

	cudaFree(Res_Layer1_Gamma);
	cudaFree(Res_Layer2_Gamma);
	cudaFree(Res_Layer3_Gamma);
	cudaFree(Res_Layer4_Gamma);
	cudaFree(Res_Layer5_Gamma);
	cudaFree(Res_Layer6_Gamma);
	cudaFree(Res_Layer7_Gamma);
	cudaFree(Res_Layer8_Gamma);
	cudaFree(Res_Layer9_Gamma);
	cudaFree(Res_Layer10_Gamma);
	cudaFree(Res_Layer11_Gamma);
	cudaFree(Res_Layer12_Gamma);
	cudaFree(Res_Layer13_Gamma);
	cudaFree(Res_Layer14_Gamma);
	cudaFree(Res_Layer15_Gamma);
	cudaFree(Res_Layer16_Gamma);
	cudaFree(Res_Layer17_Gamma);
	cudaFree(Res_Block3_Gamma);
	cudaFree(Res_Block4_Gamma);
	cudaFree(Res_Block5_Gamma);

	cudaFree(Res_Layer1_Beta);
	cudaFree(Res_Layer2_Beta);
	cudaFree(Res_Layer3_Beta);
	cudaFree(Res_Layer4_Beta);
	cudaFree(Res_Layer5_Beta);
	cudaFree(Res_Layer6_Beta);
	cudaFree(Res_Layer7_Beta);
	cudaFree(Res_Layer8_Beta);
	cudaFree(Res_Layer9_Beta);
	cudaFree(Res_Layer10_Beta);
	cudaFree(Res_Layer11_Beta);
	cudaFree(Res_Layer12_Beta);
	cudaFree(Res_Layer13_Beta);
	cudaFree(Res_Layer14_Beta);
	cudaFree(Res_Layer15_Beta);
	cudaFree(Res_Layer16_Beta);
	cudaFree(Res_Layer17_Beta);
	cudaFree(Res_Block3_Beta);
	cudaFree(Res_Block4_Beta);
	cudaFree(Res_Block5_Beta);

	cudaFree(Res_mean1);
	cudaFree(Res_mean2);
	cudaFree(Res_mean3);
	cudaFree(Res_mean4);
	cudaFree(Res_mean5);
	cudaFree(Res_mean6);
	cudaFree(Res_mean7);
	cudaFree(Res_mean8);
	cudaFree(Res_mean9);
	cudaFree(Res_mean10);
	cudaFree(Res_mean11);
	cudaFree(Res_mean12);
	cudaFree(Res_mean13);
	cudaFree(Res_mean14);
	cudaFree(Res_mean15);
	cudaFree(Res_mean16);
	cudaFree(Res_mean17);
	cudaFree(Res_Block3_mean);
	cudaFree(Res_Block4_mean);
	cudaFree(Res_Block5_mean);

	cudaFree(Res_var1);
	cudaFree(Res_var2);
	cudaFree(Res_var3);
	cudaFree(Res_var4);
	cudaFree(Res_var5);
	cudaFree(Res_var6);
	cudaFree(Res_var7);
	cudaFree(Res_var8);
	cudaFree(Res_var9);
	cudaFree(Res_var10);
	cudaFree(Res_var11);
	cudaFree(Res_var12);
	cudaFree(Res_var13);
	cudaFree(Res_var14);
	cudaFree(Res_var15);
	cudaFree(Res_var16);
	cudaFree(Res_var17);
	cudaFree(Res_Block3_var);
	cudaFree(Res_Block4_var);
	cudaFree(Res_Block5_var);

	cudaFree(Res_FC_bias);
	cudaFree(Res_FC_Weights);

	cudaFree(Res_Layer3_basic);
	cudaFree(Res_Layer5_basic);
	cudaFree(Res_Layer7_basic);
	cudaFree(Res_Layer9_basic);
	cudaFree(Res_Layer11_basic);
	cudaFree(Res_Layer13_basic);
	cudaFree(Res_Layer15_basic);
	cudaFree(Res_Layer17_basic);
	cudaFree(Res_Block3_basic);
	cudaFree(Res_Block4_basic);
	cudaFree(Res_Block5_basic);
	cudaFree(Res_Layer1_bn);
	cudaFree(Res_Layer2_bn);
	cudaFree(Res_Layer3_bn);
	cudaFree(Res_Layer4_bn);
	cudaFree(Res_Layer5_bn);
	cudaFree(Res_Layer6_bn);
	cudaFree(Res_Layer7_bn);
	cudaFree(Res_Layer8_bn);
	cudaFree(Res_Layer9_bn);
	cudaFree(Res_Layer10_bn);
	cudaFree(Res_Layer11_bn);
	cudaFree(Res_Layer12_bn);
	cudaFree(Res_Layer13_bn);
	cudaFree(Res_Layer14_bn);
	cudaFree(Res_Layer15_bn);
	cudaFree(Res_Layer16_bn);
	cudaFree(Res_Layer17_bn);
	cudaFree(Res_Block3_bn);
	cudaFree(Res_Block4_bn);
	cudaFree(Res_Block5_bn);
	cudaFree(Res_Layer1_pool);
	cudaFree(Res_FC_Neurons);
	cudaFree(Res_Result_Neurons);
}

void host2gpu_vgg16(float **Vgg_Layer1_Neurons,float **Vgg_Layer2_Neurons,float **Vgg_Layer3_Neurons,float **Vgg_Layer4_Neurons,
					float **Vgg_Layer5_Neurons,float **Vgg_Layer6_Neurons,float **Vgg_Layer7_Neurons,float **Vgg_Layer8_Neurons,
					float **Vgg_Layer9_Neurons,float **Vgg_Layer10_Neurons,float **Vgg_Layer11_Neurons,float **Vgg_Layer12_Neurons,
					float **Vgg_Layer13_Neurons,float **Vgg_Layer14_Neurons,float **Vgg_Layer15_Neurons,float **Vgg_Layer16_Neurons,
                    float **Vgg_Layer1_bias,float **Vgg_Layer2_bias,float **Vgg_Layer3_bias,float **Vgg_Layer4_bias,
                    float **Vgg_Layer5_bias,float **Vgg_Layer6_bias,float **Vgg_Layer7_bias,float **Vgg_Layer8_bias,
                    float **Vgg_Layer9_bias,float **Vgg_Layer10_bias,float **Vgg_Layer11_bias,float **Vgg_Layer12_bias,
                    float **Vgg_Layer13_bias,float **Vgg_Layer14_bias,float **Vgg_Layer15_bias,float **Vgg_Layer16_bias,
                    float **Vgg_Layer1_Weights,float **Vgg_Layer2_Weights,float **Vgg_Layer3_Weights,float **Vgg_Layer4_Weights,
                    float **Vgg_Layer5_Weights,float **Vgg_Layer6_Weights,float **Vgg_Layer7_Weights,float **Vgg_Layer8_Weights,
                    float **Vgg_Layer9_Weights,float **Vgg_Layer10_Weights,float **Vgg_Layer11_Weights,float **Vgg_Layer12_Weights,
                    float **Vgg_Layer13_Weights,float **Vgg_Layer14_Weights,float **Vgg_Layer15_Weights,float **Vgg_Layer16_Weights,
                    float **Vgg_Layer2_pool,float **Vgg_Layer4_pool,float **Vgg_Layer7_pool,float **Vgg_Layer10_pool,
					float **Vgg_Layer13_pool,float **Vgg_Result_Neurons)
{
	float *Vgg_Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));
	read_parameter("data_vgg16/input_cat.txt", Vgg_Layer1_Neurons_CPU);

	float *Vgg_Layer1_bias_CPU = (float*) malloc (64 * sizeof(float)); //64
	float *Vgg_Layer2_bias_CPU = (float*) malloc (64 * sizeof(float)); //64
	float *Vgg_Layer3_bias_CPU = (float*) malloc (128 * sizeof(float)); //128
	float *Vgg_Layer4_bias_CPU = (float*) malloc (128 * sizeof(float)); //128
	float *Vgg_Layer5_bias_CPU = (float*) malloc (256 * sizeof(float)); //256
	float *Vgg_Layer6_bias_CPU = (float*) malloc (256 * sizeof(float)); //256
	float *Vgg_Layer7_bias_CPU = (float*) malloc (256 * sizeof(float)); //256
	float *Vgg_Layer8_bias_CPU = (float*) malloc (512 * sizeof(float)); //512
    float *Vgg_Layer9_bias_CPU = (float*) malloc (512 * sizeof(float)); //512
	float *Vgg_Layer10_bias_CPU = (float*) malloc (512 * sizeof(float)); //512
	float *Vgg_Layer11_bias_CPU = (float*) malloc (512 * sizeof(float)); //512
	float *Vgg_Layer12_bias_CPU = (float*) malloc (512 * sizeof(float)); //512
	float *Vgg_Layer13_bias_CPU = (float*) malloc (512 * sizeof(float)); //512
	float *Vgg_Layer14_bias_CPU = (float*) malloc (4096 * sizeof(float)); //4096
	float *Vgg_Layer15_bias_CPU = (float*) malloc (4096 * sizeof(float)); //4096
	float *Vgg_Layer16_bias_CPU = (float*) malloc (1000 * sizeof(float)); //1000

	float *Vgg_Layer1_Weights_CPU = (float*) malloc (64*3*3*3 * sizeof(float)); //64*3*3*3 = 1,728
	float *Vgg_Layer2_Weights_CPU = (float*) malloc (64*3*3*64 * sizeof(float)); //64*3*3*64 = 36,864
	float *Vgg_Layer3_Weights_CPU = (float*) malloc (128*3*3*64 * sizeof(float)); //128*3*3*64 = 73,728
	float *Vgg_Layer4_Weights_CPU = (float*) malloc (128*3*3*128 * sizeof(float)); //128*3*3*128 = 147,456
	float *Vgg_Layer5_Weights_CPU = (float*) malloc (256*3*3*128 * sizeof(float)); //256*3*3*128 = 294,912
	float *Vgg_Layer6_Weights_CPU = (float*) malloc (256*3*3*256 * sizeof(float)); //256*3*3*256 = 589,824
	float *Vgg_Layer7_Weights_CPU = (float*) malloc (256*3*3*256 * sizeof(float)); //256*3*3*256 = 589,824
	float *Vgg_Layer8_Weights_CPU = (float*) malloc (512*3*3*256 * sizeof(float)); //512*3*3*256 = 1,179,648
    float *Vgg_Layer9_Weights_CPU = (float*) malloc (512*3*3*512 * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Vgg_Layer10_Weights_CPU = (float*) malloc (512*3*3*512 * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Vgg_Layer11_Weights_CPU = (float*) malloc (512*3*3*512 * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Vgg_Layer12_Weights_CPU = (float*) malloc (512*3*3*512 * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Vgg_Layer13_Weights_CPU = (float*) malloc (512*3*3*512 * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Vgg_Layer14_Weights_CPU = (float*) malloc (4096*512*7*7 * sizeof(float)); //4096*512*7*7 = 102,760,448
	float *Vgg_Layer15_Weights_CPU = (float*) malloc (4096*4096 * sizeof(float)); //4096*4096 = 16,777,216
	float *Vgg_Layer16_Weights_CPU = (float*) malloc (1000*4096 * sizeof(float)); //1000*4096 = 4,096,000

	read_parameter("data_vgg16/bias1.txt", Vgg_Layer1_bias_CPU);
	read_parameter("data_vgg16/bias2.txt", Vgg_Layer2_bias_CPU);
	read_parameter("data_vgg16/bias3.txt", Vgg_Layer3_bias_CPU);
	read_parameter("data_vgg16/bias4.txt", Vgg_Layer4_bias_CPU);
	read_parameter("data_vgg16/bias5.txt", Vgg_Layer5_bias_CPU);
	read_parameter("data_vgg16/bias6.txt", Vgg_Layer6_bias_CPU);
	read_parameter("data_vgg16/bias7.txt", Vgg_Layer7_bias_CPU);
	read_parameter("data_vgg16/bias8.txt", Vgg_Layer8_bias_CPU);
    read_parameter("data_vgg16/bias9.txt", Vgg_Layer9_bias_CPU);
	read_parameter("data_vgg16/bias10.txt", Vgg_Layer10_bias_CPU);
	read_parameter("data_vgg16/bias11.txt", Vgg_Layer11_bias_CPU);
	read_parameter("data_vgg16/bias12.txt", Vgg_Layer12_bias_CPU);
	read_parameter("data_vgg16/bias13.txt", Vgg_Layer13_bias_CPU);
	read_parameter("data_vgg16/bias14.txt", Vgg_Layer14_bias_CPU);
	read_parameter("data_vgg16/bias15.txt", Vgg_Layer15_bias_CPU);
	read_parameter("data_vgg16/bias16.txt", Vgg_Layer16_bias_CPU);

	read_parameter("data_vgg16/conv1.txt", Vgg_Layer1_Weights_CPU);
	read_parameter("data_vgg16/conv2.txt", Vgg_Layer2_Weights_CPU);
	read_parameter("data_vgg16/conv3.txt", Vgg_Layer3_Weights_CPU);
	read_parameter("data_vgg16/conv4.txt", Vgg_Layer4_Weights_CPU);
	read_parameter("data_vgg16/conv5.txt", Vgg_Layer5_Weights_CPU);
	read_parameter("data_vgg16/conv6.txt", Vgg_Layer6_Weights_CPU);
	read_parameter("data_vgg16/conv7.txt", Vgg_Layer7_Weights_CPU);
	read_parameter("data_vgg16/conv8.txt", Vgg_Layer8_Weights_CPU);
 	read_parameter("data_vgg16/conv9.txt", Vgg_Layer9_Weights_CPU);
	read_parameter("data_vgg16/conv10.txt", Vgg_Layer10_Weights_CPU);
	read_parameter("data_vgg16/conv11.txt", Vgg_Layer11_Weights_CPU);
	read_parameter("data_vgg16/conv12.txt", Vgg_Layer12_Weights_CPU);
	read_parameter("data_vgg16/conv13.txt", Vgg_Layer13_Weights_CPU);
	read_parameter("data_vgg16/fc14.txt", Vgg_Layer14_Weights_CPU);
	read_parameter("data_vgg16/fc15.txt", Vgg_Layer15_Weights_CPU);
	read_parameter("data_vgg16/fc16.txt", Vgg_Layer16_Weights_CPU);

    float *Vgg_Layer1_Neurons_data;
	float *Vgg_Layer1_bias_data, *Vgg_Layer2_bias_data, *Vgg_Layer3_bias_data, *Vgg_Layer4_bias_data,
			*Vgg_Layer5_bias_data, *Vgg_Layer6_bias_data, *Vgg_Layer7_bias_data, *Vgg_Layer8_bias_data,
			*Vgg_Layer9_bias_data, *Vgg_Layer10_bias_data, *Vgg_Layer11_bias_data, *Vgg_Layer12_bias_data,
			*Vgg_Layer13_bias_data, *Vgg_Layer14_bias_data, *Vgg_Layer15_bias_data, *Vgg_Layer16_bias_data;
	float *Vgg_Layer1_Weights_data, *Vgg_Layer2_Weights_data, *Vgg_Layer3_Weights_data, *Vgg_Layer4_Weights_data, 
			*Vgg_Layer5_Weights_data, *Vgg_Layer6_Weights_data, *Vgg_Layer7_Weights_data, *Vgg_Layer8_Weights_data,
			*Vgg_Layer9_Weights_data, *Vgg_Layer10_Weights_data, *Vgg_Layer11_Weights_data, *Vgg_Layer12_Weights_data, 
			*Vgg_Layer13_Weights_data, *Vgg_Layer14_Weights_data, *Vgg_Layer15_Weights_data, *Vgg_Layer16_Weights_data;
	
	cudaMalloc((void**) &Vgg_Layer1_Neurons_data, INPUT_SIZE * sizeof(float)); //224*224*3
	cudaMalloc((void**) &Vgg_Layer1_bias_data, 64 * sizeof(float)); //64
	cudaMalloc((void**) &Vgg_Layer1_Weights_data, (64*(3*3*3)) * sizeof(float)); //64*3*3*3 = 1728
	cudaMalloc((void**) &Vgg_Layer2_bias_data, 64 * sizeof(float)); //64
	cudaMalloc((void**) &Vgg_Layer2_Weights_data, (64*(3*3*64)) * sizeof(float)); //64*3*3*64 = 36864
	cudaMalloc((void**) &Vgg_Layer3_bias_data, 128 * sizeof(float)); //128
	cudaMalloc((void**) &Vgg_Layer3_Weights_data, (128*(3*3*64)) * sizeof(float)); //128*3*3*64 = 73728
	cudaMalloc((void**) &Vgg_Layer4_bias_data, 128 * sizeof(float)); //128
	cudaMalloc((void**) &Vgg_Layer4_Weights_data, (128*(3*3*128)) * sizeof(float)); //128*3*3*128 = 147456
	cudaMalloc((void**) &Vgg_Layer5_bias_data, 256 * sizeof(float)); //256
	cudaMalloc((void**) &Vgg_Layer5_Weights_data, (256*(3*3*128)) * sizeof(float)); //256*3*3*128 = 294912
	cudaMalloc((void**) &Vgg_Layer6_bias_data, 256 * sizeof(float)); //256
	cudaMalloc((void**) &Vgg_Layer6_Weights_data, (256*(3*3*256)) * sizeof(float)); //256*3*3*256 = 589824
	cudaMalloc((void**) &Vgg_Layer7_bias_data, 256 * sizeof(float)); //256
	cudaMalloc((void**) &Vgg_Layer7_Weights_data, (256*(3*3*256)) * sizeof(float)); //256*3*3*256 = 589824
	cudaMalloc((void**) &Vgg_Layer8_bias_data, 512 * sizeof(float)); //512
	cudaMalloc((void**) &Vgg_Layer8_Weights_data, (512*(3*3*256)) * sizeof(float)); //512*3*3*256 = 1179648
	cudaMalloc((void**) &Vgg_Layer9_bias_data, 512 * sizeof(float)); //512
	cudaMalloc((void**) &Vgg_Layer9_Weights_data, (512*(3*3*512)) * sizeof(float)); //512*3*3*512 = 2359296
	cudaMalloc((void**) &Vgg_Layer10_bias_data, 512 * sizeof(float)); //512
	cudaMalloc((void**) &Vgg_Layer10_Weights_data, (512*(3*3*512)) * sizeof(float)); //512*3*3*512 = 2359296
	cudaMalloc((void**) &Vgg_Layer11_bias_data, 512 * sizeof(float)); //512
	cudaMalloc((void**) &Vgg_Layer11_Weights_data, (512*(3*3*512)) * sizeof(float)); //512*3*3*512 = 2359296
	cudaMalloc((void**) &Vgg_Layer12_bias_data, 512 * sizeof(float)); //512
	cudaMalloc((void**) &Vgg_Layer12_Weights_data, (512*(3*3*512)) * sizeof(float)); //512*3*3*512 = 2359296
	cudaMalloc((void**) &Vgg_Layer13_bias_data, 512 * sizeof(float)); //256
	cudaMalloc((void**) &Vgg_Layer13_Weights_data, (512*(3*3*512)) * sizeof(float)); //512*3*3*512 = 2359296
	cudaMalloc((void**) &Vgg_Layer14_bias_data, 4096 * sizeof(float)); //4096
	cudaMalloc((void**) &Vgg_Layer14_Weights_data, (4096*(512*(7*7))) * sizeof(float)); //4096*512*7*7
	cudaMalloc((void**) &Vgg_Layer15_bias_data, 4096 * sizeof(float)); //4096
	cudaMalloc((void**) &Vgg_Layer15_Weights_data, (4096*4096) * sizeof(float)); //4096*4096
	cudaMalloc((void**) &Vgg_Layer16_bias_data, 1000 * sizeof(float)); //1000
	cudaMalloc((void**) &Vgg_Layer16_Weights_data, (1000*4096) * sizeof(float)); //1000*4096

	cudaMemcpy(Vgg_Layer1_Neurons_data, Vgg_Layer1_Neurons_CPU, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer1_bias_data, Vgg_Layer1_bias_CPU, 64 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer1_Weights_data, Vgg_Layer1_Weights_CPU, (64*(3*3*3)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer2_bias_data, Vgg_Layer2_bias_CPU, 64 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer2_Weights_data, Vgg_Layer2_Weights_CPU, (64*(3*3*64)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer3_bias_data, Vgg_Layer3_bias_CPU, 128 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer3_Weights_data, Vgg_Layer3_Weights_CPU, (128*(3*3*64)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer4_bias_data, Vgg_Layer4_bias_CPU, 64 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer4_Weights_data, Vgg_Layer4_Weights_CPU, (64*(3*3*128)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer5_bias_data, Vgg_Layer5_bias_CPU, 256 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer5_Weights_data, Vgg_Layer5_Weights_CPU, (256*(3*3*128)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer6_bias_data, Vgg_Layer6_bias_CPU, 256 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer6_Weights_data, Vgg_Layer6_Weights_CPU, (256*(3*3*256)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer7_bias_data, Vgg_Layer7_bias_CPU, 256 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer7_Weights_data, Vgg_Layer7_Weights_CPU, (256*(3*3*256)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer8_bias_data, Vgg_Layer8_bias_CPU, 512 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer8_Weights_data, Vgg_Layer8_Weights_CPU, (512*(3*3*256)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer9_bias_data, Vgg_Layer9_bias_CPU, 512 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer9_Weights_data, Vgg_Layer9_Weights_CPU, (512*(3*3*512)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer10_bias_data, Vgg_Layer10_bias_CPU, 512 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer10_Weights_data, Vgg_Layer10_Weights_CPU, (512*(3*3*512)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer11_bias_data, Vgg_Layer11_bias_CPU, 512 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer11_Weights_data, Vgg_Layer11_Weights_CPU, (512*(3*3*512)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer12_bias_data, Vgg_Layer12_bias_CPU, 512 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer12_Weights_data, Vgg_Layer12_Weights_CPU, (512*(3*3*512)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer13_bias_data, Vgg_Layer13_bias_CPU, 512 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer13_Weights_data, Vgg_Layer13_Weights_CPU, (512*(3*3*512)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer14_bias_data, Vgg_Layer14_bias_CPU, 4096 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer14_Weights_data, Vgg_Layer14_Weights_CPU, (4096*(512*(7*7))) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer15_bias_data, Vgg_Layer15_bias_CPU, 4096 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer15_Weights_data, Vgg_Layer15_Weights_CPU, (4096*4096) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer16_bias_data, Vgg_Layer16_bias_CPU, 1000 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vgg_Layer16_Weights_data, Vgg_Layer16_Weights_CPU, (1000*4096) * sizeof(float), cudaMemcpyHostToDevice);

	*Vgg_Layer1_Neurons = Vgg_Layer1_Neurons_data;

	*Vgg_Layer1_bias = Vgg_Layer1_bias_data;
	*Vgg_Layer2_bias = Vgg_Layer2_bias_data;
	*Vgg_Layer3_bias = Vgg_Layer3_bias_data;
	*Vgg_Layer4_bias = Vgg_Layer4_bias_data;
	*Vgg_Layer5_bias = Vgg_Layer5_bias_data;
	*Vgg_Layer6_bias = Vgg_Layer6_bias_data;
	*Vgg_Layer7_bias = Vgg_Layer7_bias_data;
	*Vgg_Layer8_bias = Vgg_Layer8_bias_data;
	*Vgg_Layer9_bias = Vgg_Layer9_bias_data;
	*Vgg_Layer10_bias = Vgg_Layer10_bias_data;
	*Vgg_Layer11_bias = Vgg_Layer11_bias_data;
	*Vgg_Layer12_bias = Vgg_Layer12_bias_data;
	*Vgg_Layer13_bias = Vgg_Layer13_bias_data;
	*Vgg_Layer14_bias = Vgg_Layer14_bias_data;
	*Vgg_Layer15_bias = Vgg_Layer15_bias_data;
	*Vgg_Layer16_bias = Vgg_Layer16_bias_data;

	*Vgg_Layer1_Weights = Vgg_Layer1_Weights_data;
	*Vgg_Layer2_Weights = Vgg_Layer2_Weights_data;
	*Vgg_Layer3_Weights = Vgg_Layer3_Weights_data;
	*Vgg_Layer4_Weights = Vgg_Layer4_Weights_data;
	*Vgg_Layer5_Weights = Vgg_Layer5_Weights_data;
	*Vgg_Layer6_Weights = Vgg_Layer6_Weights_data;
	*Vgg_Layer7_Weights = Vgg_Layer7_Weights_data;
	*Vgg_Layer8_Weights = Vgg_Layer8_Weights_data;
	*Vgg_Layer9_Weights = Vgg_Layer9_Weights_data;
	*Vgg_Layer10_Weights = Vgg_Layer10_Weights_data;
	*Vgg_Layer11_Weights = Vgg_Layer11_Weights_data;
	*Vgg_Layer12_Weights = Vgg_Layer12_Weights_data;
	*Vgg_Layer13_Weights = Vgg_Layer13_Weights_data;
	*Vgg_Layer14_Weights = Vgg_Layer14_Weights_data;
	*Vgg_Layer15_Weights = Vgg_Layer15_Weights_data;
	*Vgg_Layer16_Weights = Vgg_Layer16_Weights_data;

	free(Vgg_Layer1_Neurons_CPU);
	free(Vgg_Layer1_bias_CPU);
	free(Vgg_Layer2_bias_CPU);
	free(Vgg_Layer3_bias_CPU);
	free(Vgg_Layer4_bias_CPU);
	free(Vgg_Layer5_bias_CPU);
	free(Vgg_Layer6_bias_CPU);
	free(Vgg_Layer7_bias_CPU);
	free(Vgg_Layer8_bias_CPU);
	free(Vgg_Layer9_bias_CPU);
	free(Vgg_Layer10_bias_CPU);
	free(Vgg_Layer11_bias_CPU);
	free(Vgg_Layer12_bias_CPU);
	free(Vgg_Layer13_bias_CPU);
	free(Vgg_Layer14_bias_CPU);
	free(Vgg_Layer15_bias_CPU);
	free(Vgg_Layer16_bias_CPU);

    free(Vgg_Layer1_Weights_CPU);
    free(Vgg_Layer2_Weights_CPU);
    free(Vgg_Layer3_Weights_CPU);
    free(Vgg_Layer4_Weights_CPU);
    free(Vgg_Layer5_Weights_CPU);
    free(Vgg_Layer6_Weights_CPU);
    free(Vgg_Layer7_Weights_CPU);
    free(Vgg_Layer8_Weights_CPU);
	free(Vgg_Layer9_Weights_CPU);
    free(Vgg_Layer10_Weights_CPU);
    free(Vgg_Layer11_Weights_CPU);
    free(Vgg_Layer12_Weights_CPU);
    free(Vgg_Layer13_Weights_CPU);
    free(Vgg_Layer14_Weights_CPU);
    free(Vgg_Layer15_Weights_CPU);
    free(Vgg_Layer16_Weights_CPU);

	float *Vgg_Layer2_Neurons_data; 
	cudaMalloc((void**) &Vgg_Layer2_Neurons_data, (64*224*224) * sizeof(float)); //64*224*224
	*Vgg_Layer2_Neurons = Vgg_Layer2_Neurons_data;

    float *Vgg_Layer2_pool_data;
    cudaMalloc((void**) &Vgg_Layer2_pool_data, (64*224*224) * sizeof(float)); //64*224*224
	*Vgg_Layer2_pool = Vgg_Layer2_pool_data;

    float *Vgg_Layer3_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer3_Neurons_data, (64*112*112) * sizeof(float)); //64*112*112
	*Vgg_Layer3_Neurons = Vgg_Layer3_Neurons_data;

    float *Vgg_Layer4_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer4_Neurons_data, (128*112*112) * sizeof(float)); //128*112*112
	*Vgg_Layer4_Neurons = Vgg_Layer4_Neurons_data;

    float *Vgg_Layer4_pool_data;
    cudaMalloc((void**) &Vgg_Layer4_pool_data, (128*112*112) * sizeof(float)); //128*112*112
	*Vgg_Layer4_pool = Vgg_Layer4_pool_data;
	
    float *Vgg_Layer5_Neurons_data;
	cudaMalloc((void**) &Vgg_Layer5_Neurons_data, (128*56*56) * sizeof(float)); //128*56*56
	*Vgg_Layer5_Neurons = Vgg_Layer5_Neurons_data;

    float *Vgg_Layer6_Neurons_data;
   	cudaMalloc((void**) &Vgg_Layer6_Neurons_data, (256*56*56) * sizeof(float)); //256*56*56
	*Vgg_Layer6_Neurons = Vgg_Layer6_Neurons_data;

    float *Vgg_Layer7_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer7_Neurons_data, (256*56*56) * sizeof(float)); //256*56*56
	*Vgg_Layer7_Neurons = Vgg_Layer7_Neurons_data;

    float *Vgg_Layer7_pool_data;
    cudaMalloc((void**) &Vgg_Layer7_pool_data, (256*56*56) * sizeof(float)); //256*56*56
	*Vgg_Layer7_pool = Vgg_Layer7_pool_data;

    float *Vgg_Layer8_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer8_Neurons_data, (256*28*28) * sizeof(float)); //256*28*28
	*Vgg_Layer8_Neurons = Vgg_Layer8_Neurons_data;

    float *Vgg_Layer9_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer9_Neurons_data, (512*28*28) * sizeof(float)); //512*28*28
	*Vgg_Layer9_Neurons = Vgg_Layer9_Neurons_data;

    float *Vgg_Layer10_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer10_Neurons_data, (512*28*28) * sizeof(float)); //512*28*28
	*Vgg_Layer10_Neurons = Vgg_Layer10_Neurons_data;

    float *Vgg_Layer10_pool_data;
    cudaMalloc((void**) &Vgg_Layer10_pool_data, (512*28*28) * sizeof(float)); //512*28*28
	*Vgg_Layer10_pool = Vgg_Layer10_pool_data;

    float *Vgg_Layer11_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer11_Neurons_data, (512*14*14) * sizeof(float)); //512*14*14
	*Vgg_Layer11_Neurons = Vgg_Layer11_Neurons_data;

    float *Vgg_Layer12_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer12_Neurons_data, (512*14*14) * sizeof(float)); //512*14*14 
	*Vgg_Layer12_Neurons = Vgg_Layer12_Neurons_data;

    float *Vgg_Layer13_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer13_Neurons_data, (512*14*14) * sizeof(float)); //512*14*14
	*Vgg_Layer13_Neurons = Vgg_Layer13_Neurons_data;

    float *Vgg_Layer13_pool_data;
    cudaMalloc((void**) &Vgg_Layer13_pool_data, (512*14*14) * sizeof(float)); //512*14*14
	*Vgg_Layer13_pool = Vgg_Layer13_pool_data;

    float *Vgg_Layer14_Neurons_data;
    cudaMalloc((void**) &Vgg_Layer14_Neurons_data, (512*7*7) * sizeof(float)); //512*7*7
	*Vgg_Layer14_Neurons = Vgg_Layer14_Neurons_data;

    float *Vgg_Layer15_Neurons_data;
	cudaMalloc((void**) &Vgg_Layer15_Neurons_data, 4096 * sizeof(float)); //4096
	*Vgg_Layer15_Neurons = Vgg_Layer15_Neurons_data;

    float *Vgg_Layer16_Neurons_data;
	cudaMalloc((void**) &Vgg_Layer16_Neurons_data, 4096 * sizeof(float)); //4096
	*Vgg_Layer16_Neurons = Vgg_Layer16_Neurons_data;

    float *Vgg_Result_Neurons_data;
	cudaMalloc((void**) &Vgg_Result_Neurons_data, 1000 * sizeof(float)); //1000
	*Vgg_Result_Neurons = Vgg_Result_Neurons_data;
}



void vgg_first_conv(float *Vgg_Layer1_bias,float *Vgg_Layer1_Neurons,float *Vgg_Layer1_Weights,float *Vgg_Layer2_Neurons)
{
	dim3 Block1_Block(64,32,32);
    dim3 Block_Thread(7,7);
	first<<<Block1_Block,Block_Thread>>>(Vgg_Layer1_bias,Vgg_Layer1_Neurons,Vgg_Layer1_Weights,Vgg_Layer2_Neurons,224,224,1,1,3,3,true,true);
}

void vgg_second_conv(float *Vgg_Layer2_bias,float *Vgg_Layer2_Neurons,float *Vgg_Layer2_Weights,float *Vgg_Layer2_pool)
{
	dim3 Block1_Block(64,32,32);
    dim3 Block_Thread(7,7);
	Vgg_conv<<<Block1_Block,Block_Thread>>>(Vgg_Layer2_bias,Vgg_Layer2_Neurons,Vgg_Layer2_Weights,Vgg_Layer2_pool,224,224,1,1,3,64,true,true);
}

void vgg_second_pool(float *Vgg_Layer2_pool,float *Vgg_Layer3_Neurons)
{
	dim3 Block1_Pool_Block(64,16,16);
	dim3 Block_Thread(7,7);
    max<<<Block1_Pool_Block,Block_Thread>>>(Vgg_Layer2_pool,Vgg_Layer3_Neurons,224,112,2,0,2);
}

void vgg_third_conv(float *Vgg_Layer3_bias,float *Vgg_Layer3_Neurons,float *Vgg_Layer3_Weights,float *Vgg_Layer4_Neurons)
{
	dim3 Block2_Block(128,16,16);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block2_Block,Block_Thread>>>(Vgg_Layer3_bias,Vgg_Layer3_Neurons,Vgg_Layer3_Weights,Vgg_Layer4_Neurons,112,112,1,1,3,64,true,true);
}

void vgg_fourth_conv(float *Vgg_Layer4_bias,float *Vgg_Layer4_Neurons,float *Vgg_Layer4_Weights,float *Vgg_Layer4_pool)
{
	dim3 Block2_Block(128,16,16);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block2_Block,Block_Thread>>>(Vgg_Layer4_bias,Vgg_Layer4_Neurons,Vgg_Layer4_Weights,Vgg_Layer4_pool,112,112,1,1,3,128,true,true);
}

void vgg_fourth_pool(float *Vgg_Layer4_pool,float *Vgg_Layer5_Neurons)
{
	dim3 Block2_Pool_Block(128,8,8);
	dim3 Block_Thread(7,7);
    max<<<Block2_Pool_Block,Block_Thread>>>(Vgg_Layer4_pool,Vgg_Layer5_Neurons,112,56,2,0,2);
}

void vgg_fifth_conv(float *Vgg_Layer5_bias,float *Vgg_Layer5_Neurons,float *Vgg_Layer5_Weights,float *Vgg_Layer6_Neurons)
{
	dim3 Block3_Block(256,8,8);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block3_Block,Block_Thread>>>(Vgg_Layer5_bias,Vgg_Layer5_Neurons,Vgg_Layer5_Weights,Vgg_Layer6_Neurons,56,56,1,1,3,128,true,true);
}

void vgg_sixth_conv(float *Vgg_Layer6_bias,float *Vgg_Layer6_Neurons,float *Vgg_Layer6_Weights,float *Vgg_Layer7_Neurons)
{
	dim3 Block3_Block(256,8,8);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block3_Block,Block_Thread>>>(Vgg_Layer6_bias,Vgg_Layer6_Neurons,Vgg_Layer6_Weights,Vgg_Layer7_Neurons,56,56,1,1,3,256,true,true);
}

void vgg_seventh_conv(float *Vgg_Layer7_bias,float *Vgg_Layer7_Neurons,float *Vgg_Layer7_Weights,float *Vgg_Layer7_pool)
{
	dim3 Block3_Block(256,8,8);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block3_Block,Block_Thread>>>(Vgg_Layer7_bias,Vgg_Layer7_Neurons,Vgg_Layer7_Weights,Vgg_Layer7_pool,56,56,1,1,3,256,true,true);
}

void vgg_seventh_pool(float *Vgg_Layer7_pool,float *Vgg_Layer8_Neurons)
{
	dim3 Block3_Pool_Block(256,4,4);
	dim3 Block_Thread(7,7);
    max<<<Block3_Pool_Block,Block_Thread>>>(Vgg_Layer7_pool,Vgg_Layer8_Neurons,56,28,2,0,2);
}

void vgg_eighth_conv(float *Vgg_Layer8_bias,float *Vgg_Layer8_Neurons,float *Vgg_Layer8_Weights,float *Vgg_Layer9_Neurons)
{
	dim3 Block4_Block(512,4,4);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block4_Block,Block_Thread>>>(Vgg_Layer8_bias,Vgg_Layer8_Neurons,Vgg_Layer8_Weights,Vgg_Layer9_Neurons,28,28,1,1,3,256,true,true);
}

void vgg_ninth_conv(float *Vgg_Layer9_bias,float *Vgg_Layer9_Neurons,float *Vgg_Layer9_Weights,float *Vgg_Layer10_Neurons)
{
	dim3 Block4_Block(512,4,4);
	dim3 Block_Thread(7,7);
    Vgg_conv<<<Block4_Block,Block_Thread>>>(Vgg_Layer9_bias,Vgg_Layer9_Neurons,Vgg_Layer9_Weights,Vgg_Layer10_Neurons,28,28,1,1,3,512,true,true);
}

void vgg_tenth_conv(float *Vgg_Layer10_bias,float *Vgg_Layer10_Neurons,float *Vgg_Layer10_Weights,float *Vgg_Layer10_pool)
{
	dim3 Block4_Block(512,4,4);
	dim3 Block_Thread(7,7);
    Vgg_conv<<<Block4_Block,Block_Thread>>>(Vgg_Layer10_bias,Vgg_Layer10_Neurons,Vgg_Layer10_Weights,Vgg_Layer10_pool,28,28,1,1,3,512,true,true);
}

void vgg_tenth_pool(float *Vgg_Layer10_pool,float *Vgg_Layer11_Neurons)
{
	dim3 Block4_Pool_Block(512,2,2);
	dim3 Block_Thread(7,7);	
    max<<<Block4_Pool_Block,Block_Thread>>>(Vgg_Layer10_pool,Vgg_Layer11_Neurons,28,14,2,0,2);
}

void vgg_eleventh_conv(float *Vgg_Layer11_bias,float *Vgg_Layer11_Neurons,float *Vgg_Layer11_Weights,float *Vgg_Layer12_Neurons)
{
	dim3 Block5_Block(512,2,2);
	dim3 Block_Thread(7,7);
	Vgg_conv<<<Block5_Block,Block_Thread>>>(Vgg_Layer11_bias,Vgg_Layer11_Neurons,Vgg_Layer11_Weights,Vgg_Layer12_Neurons,14,14,1,1,3,512,true,true);
}

void vgg_twelfth_conv(float *Vgg_Layer12_bias,float *Vgg_Layer12_Neurons,float *Vgg_Layer12_Weights,float *Vgg_Layer13_Neurons)
{
	dim3 Block5_Block(512,2,2);
	dim3 Block_Thread(7,7);
    Vgg_conv<<<Block5_Block,Block_Thread>>>(Vgg_Layer12_bias,Vgg_Layer12_Neurons,Vgg_Layer12_Weights,Vgg_Layer13_Neurons,14,14,1,1,3,512,true,true);
}

void vgg_thirteenth_conv(float *Vgg_Layer13_bias,float *Vgg_Layer13_Neurons,float *Vgg_Layer13_Weights,float *Vgg_Layer13_pool)
{
	dim3 Block5_Block(512,2,2);
	dim3 Block_Thread(7,7);
    Vgg_conv<<<Block5_Block,Block_Thread>>>(Vgg_Layer13_bias,Vgg_Layer13_Neurons,Vgg_Layer13_Weights,Vgg_Layer13_pool,14,14,1,1,3,512,true,true);
}

void vgg_thirteenth_pool(float *Vgg_Layer13_pool,float *Vgg_Layer14_Neurons)
{
	dim3 Block5_Pool_Block(512,1,1);
	dim3 Block_Thread(7,7);
    max<<<Block5_Pool_Block,Block_Thread>>>(Vgg_Layer13_pool,Vgg_Layer14_Neurons,14,7,2,0,2);
}

void vgg_first_fc(float *Vgg_Layer14_bias,float *Vgg_Layer14_Neurons,float *Vgg_Layer14_Weights,float *Vgg_Layer15_Neurons)
{
	dim3 FC1_Block(4096,1,1);
	dim3 FC1_Thread(1,1);
	fc<<<FC1_Block,FC1_Thread>>>(Vgg_Layer14_bias,Vgg_Layer14_Neurons,Vgg_Layer14_Weights,Vgg_Layer15_Neurons,(7*7*512),true);
}

void vgg_second_fc(float *Vgg_Layer15_bias,float *Vgg_Layer15_Neurons,float *Vgg_Layer15_Weights,float *Vgg_Layer16_Neurons)
{
	dim3 FC2_Block(4096,1,1);
	dim3 FC2_Thread(1,1);
	fc<<<FC2_Block,FC2_Thread>>>(Vgg_Layer15_bias,Vgg_Layer15_Neurons,Vgg_Layer15_Weights,Vgg_Layer16_Neurons,4096,true);
}

void vgg_third_fc(float *Vgg_Layer16_bias,float *Vgg_Layer16_Neurons,float *Vgg_Layer16_Weights,float *Vgg_Result_Neurons)
{
	dim3 FC3_Block(1000,1,1);
	dim3 FC3_Thread(1,1);
	fc<<<FC3_Block,FC3_Thread>>>(Vgg_Layer16_bias,Vgg_Layer16_Neurons,Vgg_Layer16_Weights,Vgg_Result_Neurons,4096,false);

	float *Vgg_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
	cudaMemcpy(Vgg_Result_Neurons_CPU, Vgg_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);


	float max1 = 0.0;
	int index1 = 0; 
	for(int i = 0; i < 1000; i++){
		if(max1 < Vgg_Result_Neurons_CPU[i]){
			max1 = Vgg_Result_Neurons_CPU[i];	
			index1 = i;
		}
	}
	
	int line_count1 = 0;
	char buffer[1000];
	FILE *list1 = fopen("common/imagenet1000_clsidx_to_labels.txt","rt");
	while(fgets(buffer, 1000, list1) != NULL){
		line_count1++;
		if(line_count1 == (index1+1)){
			// printf("\n---Vgg16 Result---");
			// printf("\nClass ID: %d\nClass Name: %sProbability: %f\n", index1, buffer, max1);
			printf("\nVgg16: %d, %s", index1, buffer);
			break;
		}
	}
	fclose(list1);

	free(Vgg_Result_Neurons_CPU);
}

void free_vgg16(float *Vgg_Layer1_Neurons,float *Vgg_Layer2_Neurons,float *Vgg_Layer3_Neurons,float *Vgg_Layer4_Neurons,
					float *Vgg_Layer5_Neurons,float *Vgg_Layer6_Neurons,float *Vgg_Layer7_Neurons,float *Vgg_Layer8_Neurons,
					float *Vgg_Layer9_Neurons,float *Vgg_Layer10_Neurons,float *Vgg_Layer11_Neurons,float *Vgg_Layer12_Neurons,
					float *Vgg_Layer13_Neurons,float *Vgg_Layer14_Neurons,float *Vgg_Layer15_Neurons,float *Vgg_Layer16_Neurons,
                    float *Vgg_Layer1_bias,float *Vgg_Layer2_bias,float *Vgg_Layer3_bias,float *Vgg_Layer4_bias,
                    float *Vgg_Layer5_bias,float *Vgg_Layer6_bias,float *Vgg_Layer7_bias,float *Vgg_Layer8_bias,
                    float *Vgg_Layer9_bias,float *Vgg_Layer10_bias,float *Vgg_Layer11_bias,float *Vgg_Layer12_bias,
                    float *Vgg_Layer13_bias,float *Vgg_Layer14_bias,float *Vgg_Layer15_bias,float *Vgg_Layer16_bias,
                    float *Vgg_Layer1_Weights,float *Vgg_Layer2_Weights,float *Vgg_Layer3_Weights,float *Vgg_Layer4_Weights,
                    float *Vgg_Layer5_Weights,float *Vgg_Layer6_Weights,float *Vgg_Layer7_Weights,float *Vgg_Layer8_Weights,
                    float *Vgg_Layer9_Weights,float *Vgg_Layer10_Weights,float *Vgg_Layer11_Weights,float *Vgg_Layer12_Weights,
                    float *Vgg_Layer13_Weights,float *Vgg_Layer14_Weights,float *Vgg_Layer15_Weights,float *Vgg_Layer16_Weights,
                    float *Vgg_Layer2_pool,float *Vgg_Layer4_pool,float *Vgg_Layer7_pool,float *Vgg_Layer10_pool,
					float *Vgg_Layer13_pool,float *Vgg_Result_Neurons)
{
	cudaFree(Vgg_Layer1_Neurons);
    cudaFree(Vgg_Layer2_Neurons);
	cudaFree(Vgg_Layer3_Neurons);
	cudaFree(Vgg_Layer4_Neurons);
	cudaFree(Vgg_Layer5_Neurons);
	cudaFree(Vgg_Layer6_Neurons);
	cudaFree(Vgg_Layer7_Neurons);
	cudaFree(Vgg_Layer8_Neurons);
	cudaFree(Vgg_Layer9_Neurons);
	cudaFree(Vgg_Layer10_Neurons);
	cudaFree(Vgg_Layer11_Neurons);
	cudaFree(Vgg_Layer12_Neurons);
	cudaFree(Vgg_Layer13_Neurons);
	cudaFree(Vgg_Layer14_Neurons);
	cudaFree(Vgg_Layer15_Neurons);
	cudaFree(Vgg_Layer16_Neurons);

	cudaFree(Vgg_Layer1_bias);
	cudaFree(Vgg_Layer2_bias);
	cudaFree(Vgg_Layer3_bias);
	cudaFree(Vgg_Layer4_bias);
	cudaFree(Vgg_Layer5_bias);
	cudaFree(Vgg_Layer6_bias);
	cudaFree(Vgg_Layer7_bias);
	cudaFree(Vgg_Layer8_bias);
	cudaFree(Vgg_Layer9_bias);
	cudaFree(Vgg_Layer10_bias);
	cudaFree(Vgg_Layer11_bias);
	cudaFree(Vgg_Layer12_bias);
	cudaFree(Vgg_Layer13_bias);
	cudaFree(Vgg_Layer14_bias);
	cudaFree(Vgg_Layer15_bias);
	cudaFree(Vgg_Layer16_bias);

	cudaFree(Vgg_Layer1_Weights);
	cudaFree(Vgg_Layer2_Weights);
	cudaFree(Vgg_Layer3_Weights);
	cudaFree(Vgg_Layer4_Weights);
	cudaFree(Vgg_Layer5_Weights);
	cudaFree(Vgg_Layer6_Weights);
	cudaFree(Vgg_Layer7_Weights);
	cudaFree(Vgg_Layer8_Weights);
	cudaFree(Vgg_Layer9_Weights);
	cudaFree(Vgg_Layer10_Weights);
	cudaFree(Vgg_Layer11_Weights);
	cudaFree(Vgg_Layer12_Weights);
	cudaFree(Vgg_Layer13_Weights);
	cudaFree(Vgg_Layer14_Weights);
	cudaFree(Vgg_Layer15_Weights);
	cudaFree(Vgg_Layer16_Weights);

	cudaFree(Vgg_Layer2_pool);
	cudaFree(Vgg_Layer4_pool);
	cudaFree(Vgg_Layer7_pool);
	cudaFree(Vgg_Layer10_pool);
	cudaFree(Vgg_Layer13_pool);
	cudaFree(Vgg_Result_Neurons);
}
}




