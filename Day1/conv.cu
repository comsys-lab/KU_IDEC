#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

__global__ void conv(float *bias,float *in_nu,float *in_w,float *out_nu,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,bool b,bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

	float product = 0.0;
	for(int i = 0; i < ker_channel; i++){
		for(int j = 0; j < loopr; j++){
			for(int k = 0; k < loopc; k++){
				product += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
						*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
			}
		}
	}
	if(loopc > 0 && loopr > 0){
		if(b == true)
			product += bias[num_out];

		//ReLU
		if(relu == true){
			if(product < 0)
				product = 0;
		}
		out_nu[out_position] = product;
	}
}


void initialize_data(float* data, int size, float value) {
    for (int i = 0; i < size; i++) {
        data[i] = value;
    }
}

int main() {
    // Define dimensions for the test
    int in_fm = 56;              // Input feature map size (width/height)
    int out_fm = 56;             // Output feature map size
    int ker = 3;                // Kernel size (3x3)
    int ker_channel = 3;        // Number of input channels
    int out_channels = 64;       // Number of output channels (num_out)
    int str = 1;                // Stride
    int pad = 1;                // Padding
    bool b = true;              // Apply bias
    bool relu = true;           // Apply ReLU activation

    // Host memory allocation
    int input_size = in_fm * in_fm * ker_channel;  // Size of input data
    int output_size = out_fm * out_fm * out_channels; // Size of output data
    int filter_size = ker * ker * ker_channel * out_channels; // Size of filter
    int bias_size = out_channels; // Size of bias

    float *h_in_nu = (float*)malloc(input_size * sizeof(float));
    float *h_in_w = (float*)malloc(filter_size * sizeof(float));
    float *h_out_nu = (float*)malloc(output_size * sizeof(float));
    float *h_bias = (float*)malloc(bias_size * sizeof(float));

    // Initialize host data
    initialize_data(h_in_nu, input_size, 1.0f);   // Input data initialized to 1.0
    initialize_data(h_in_w, filter_size, 0.5f);   // Filter data initialized to 0.5
    initialize_data(h_bias, bias_size, 1.0f);     // Bias initialized to 1.0

    // Device memory allocation
    float *d_in_nu, *d_in_w, *d_out_nu, *d_bias;
    cudaMalloc(&d_in_nu, input_size * sizeof(float));
    cudaMalloc(&d_in_w, filter_size * sizeof(float));
    cudaMalloc(&d_out_nu, output_size * sizeof(float));
    cudaMalloc(&d_bias, bias_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_in_nu, h_in_nu, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_w, h_in_w, filter_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(16, 16); // Block of 16x16 threads
    dim3 dimGrid(out_channels, (out_fm + dimBlock.x - 1) / dimBlock.x, (out_fm + dimBlock.y - 1) / dimBlock.y); // Grid size

    // Launch the convolution kernel
    conv<<<dimGrid, dimBlock>>>(d_bias, d_in_nu, d_in_w, d_out_nu, in_fm, out_fm, str, pad, ker, ker_channel, b, relu);

    // Copy the result from device to host
    cudaMemcpy(h_out_nu, d_out_nu, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output for verification
   

    // Free device memory
    cudaFree(d_in_nu);
    cudaFree(d_in_w);
    cudaFree(d_out_nu);
    cudaFree(d_bias);

    // Free host memory
    free(h_in_nu);
    free(h_in_w);
    free(h_out_nu);
    free(h_bias);

    std::cout << "Convolution done!" << std::endl;


    return 0;
}