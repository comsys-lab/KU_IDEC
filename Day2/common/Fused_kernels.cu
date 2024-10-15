#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void fused_first_layer(float *a_bias,float *a_in_w,float *r_in_w,
float *a_in_nu,float *r_in_nu,
float *a_out_nu,float *r_out_nu,
int alex_num,int res_num,
int a_in_fm,int a_out_fm,int a_str,int a_pad,int a_ker,int a_ker_channel,
int r_in_fm,int r_out_fm,int r_str,int r_pad,int r_ker,int r_ker_channel,
int model1_bidyz,int model1_tidxy,
int model2_bidyz,int model2_tidxy)
{
    int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

	if((row_out_block < model1_bidyz) && (col_out_block < model1_bidyz) && (row_out_thread < model1_tidxy) && (col_out_thread < model1_tidxy)){
		//// Alexnet ////
		int a_row = ((model1_tidxy*row_out_block)+(row_out_thread));
    	int a_col = ((model1_tidxy*col_out_block)+(col_out_thread));
	  
	  	int a_out_position = (a_out_fm*a_out_fm*num_out)
						+ (a_out_fm*a_row)
						+ a_col;

		//Stride
		int a_x_str = 0, a_y_str = 0;
		a_x_str = 3*(a_row*a_str-a_pad)*a_in_fm;
		a_x_str = a_x_str < 0 ? 0 : a_x_str;
		a_y_str = 3*(a_col*a_str-a_pad);
		a_y_str = a_y_str < 0 ? 0 : a_y_str;

		//Padding
		int a_x_pad = 0, a_y_pad = 0;
		int a_loopr = a_ker, a_loopc = a_ker;

		//Upper
		if(a_row*a_str < a_pad){
			a_x_pad = a_pad - a_row*a_str;
			a_loopr = a_ker - a_x_pad;
		}
		//Bottom
		if(a_row >= a_out_fm - a_pad){
			a_loopr = a_in_fm - a_x_str/(3*a_in_fm);
		}
		//Left
		if(a_col*a_str < a_pad){
			a_y_pad = a_pad - a_col*a_str;
			a_loopc = a_ker - a_y_pad;
		}
		//Right
		if(a_col >= a_out_fm - a_pad){
			a_loopc = a_in_fm - a_y_str/3;
		}

		float a_product[5] = {0.0};
		for(int i = 0; i < a_loopr; i++){
			for(int j = 0; j < a_loopc; j++){
				for(int k = 0; k < a_ker_channel; k++){
					for(int l = 0; l < alex_num; l++){
						a_product[l] += a_in_nu[i*a_in_fm*a_ker_channel + j*a_ker_channel + k + a_x_str + a_y_str]
										* a_in_w[num_out*a_ker*a_ker*a_ker_channel + i*a_ker + j + k*a_ker*a_ker + a_x_pad*a_ker + a_y_pad];
					}
				}
			}
		}
		for(int i = 0; i < alex_num; i++){
			if(a_loopc > 0 && a_loopr > 0){			
				a_product[i] += a_bias[num_out];
				//ReLU
				if(a_product[i] < 0)
					a_product[i] = 0;
			}
			a_out_nu[a_out_position] = a_product[i];
		}
	}

	if((row_out_block < model2_bidyz) && (col_out_block < model2_bidyz) && (row_out_thread < model2_tidxy) && (col_out_thread < model2_tidxy)){
		
		//// Resnet18 ////
		int r_row = ((model2_tidxy*row_out_block)+(row_out_thread));
    	int r_col = ((model2_tidxy*col_out_block)+(col_out_thread));

		int r_out_position = (r_out_fm*r_out_fm*num_out)
						+ (r_out_fm*r_row)
						+ r_col;

		//Stride
		int r_x_str = 0, r_y_str = 0;
		r_x_str = 3*(r_row*r_str-r_pad)*r_in_fm;
		r_x_str = r_x_str < 0 ? 0 : r_x_str;
		r_y_str = 3*(r_col*r_str-r_pad);
		r_y_str = r_y_str < 0 ? 0 : r_y_str;

		//Padding
		int r_x_pad = 0, r_y_pad = 0;
		int r_loopr = r_ker, r_loopc = r_ker;

		//Upper
		if(r_row*r_str < r_pad){
			r_x_pad = r_pad - r_row*r_str;
			r_loopr = r_ker - r_x_pad;
		}
		//Bottom
		if(r_row >= r_out_fm - r_pad){
			r_loopr = r_in_fm - r_x_str/(3*r_in_fm);
		}
		//Left
		if(r_col*r_str < r_pad){
			r_y_pad = r_pad - r_col*r_str;
			r_loopc = r_ker - r_y_pad;
		}
		//Right
		if(r_col >= r_out_fm - r_pad){
			r_loopc = r_in_fm -  r_y_str/3;
		}

		float r_product[5] = {0.0};
		for(int i = 0; i < r_loopr; i++){
			for(int j = 0; j < r_loopc; j++){
				for(int k = 0; k < r_ker_channel; k++){
					for(int l = 0; l < res_num; l++){
						r_product[l] += r_in_nu[i*r_in_fm*r_ker_channel + j*r_ker_channel + k + r_x_str + r_y_str]
										* r_in_w[num_out*r_ker*r_ker*r_ker_channel + i*r_ker + j + k*r_ker*r_ker + r_x_pad*r_ker + r_y_pad];
					}	
				}		
			}
		}
		for(int i = 0; i < res_num; i++){
			if(r_loopc > 0 && r_loopr > 0){
				//ReLU
				if(r_product[i] < 0)
					r_product[i] = 0;
			}
			r_out_nu[r_out_position] = r_product[i];
		}
	}
}

__global__ void fused_lrm_bn1(float *a_in_nu,float *r_in_nu,
float *a_out_nu,float *r_out_nu,
int alex_num,int res_num,
float alpha,float beta1,int local_size,int a_out_fm,
float *mean,float *var,float *gamma,float *beta,int r_out_fm,bool bn_relu,
int model1_bidx,int model1_bidyz,int model1_tidxy,
int model2_bidx,int model2_bidyz,int model2_tidxy)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

	if((blockIdx.x < model1_bidx) && (blockIdx.y < model1_bidyz) && (blockIdx.z < model1_bidyz) && (threadIdx.x < model1_tidxy) && (threadIdx.y < model1_tidxy))
	{
		int a_row = ((model1_tidxy*row_out_block)+(row_out_thread));
		int a_col = ((model1_tidxy*col_out_block)+(col_out_thread));
	
		int a_out_position = (a_out_fm*a_out_fm*num_out)
						+ (a_out_fm*a_row)
						+ a_col;
		int a_input_position = (a_out_fm*a_row) + a_col;

		int nStart = 0, nEnd = 0;
		nStart=(num_out-2) > 1 ? (num_out-2) : 1 ;
		nEnd=(num_out+2) < gridDim.x ? (num_out+2) : gridDim.x ;

		float a_sum[5] = {0.0};
		float a_result[5] = {0.0}; 
		for(int i = 0; i < alex_num; i++){
			for(int j = (nStart-1); j < (nEnd-1); j++){
				a_sum[i] += pow((a_in_nu[j*a_out_fm*a_out_fm + a_input_position]),2);
			}
			a_result[i] = (a_in_nu[a_out_position]) / (pow(1 + ((alpha/local_size) * a_sum[i]),beta1));
			a_out_nu[a_out_position] = a_result[i];
		}
	}
	if((blockIdx.x < model2_bidx) && (blockIdx.y < model2_bidyz) && (blockIdx.z < model2_bidyz) && (threadIdx.x < model2_tidxy) && (threadIdx.y < model2_tidxy))
	{
		int r_row = ((model2_tidxy*row_out_block)+(row_out_thread));
		int r_col = ((model2_tidxy*col_out_block)+(col_out_thread));
	
		int r_out_position = (r_out_fm*r_out_fm*num_out)
						+ (r_out_fm*r_row)
						+ r_col;

		float product[5] = {0.0};
		for(int i = 0; i < res_num; i++){
			product[i] = ((r_in_nu[r_out_position] - mean[num_out])/(sqrt(var[num_out] + 1e-5)))*gamma[num_out] + beta[num_out];
			// relu
			if(bn_relu == true){
				if(product[i] < 0)
					product[i] = 0;
			}
			r_out_nu[r_out_position] = product[i];
		}
	}
}

__global__ void fused_max1(float *in_nu1,float *in_nu2,
float *out_nu1,float *out_nu2,
int model_num1,int model_num2,
int in_fm1,int out_fm1,int str1,int pad1,int ker1,
int in_fm2,int out_fm2,int str2,int pad2,int ker2,
int model1_bidx,int model1_bidyz,int model1_tidxy,
int model2_bidx,int model2_bidyz,int model2_tidxy)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

	if((blockIdx.x < model1_bidx) && (blockIdx.y < model1_bidyz) && (blockIdx.z < model1_bidyz) && (threadIdx.x < model1_tidxy) && (threadIdx.y < model1_tidxy))
	{
		int row1 = ((model1_tidxy*row_out_block)+(row_out_thread));
		int col1 = ((model1_tidxy*col_out_block)+(col_out_thread));
	
		int out_position1 = (out_fm1*out_fm1*num_out)
						+ (out_fm1*row1)
						+ col1;
		//Stride
		int x_str1 = 0, y_str1 = 0;
		x_str1 = (row1*str1-pad1)*in_fm1;
		x_str1 = x_str1 < 0 ? 0 : x_str1;
		y_str1 = col1*str1-pad1;
		y_str1 = y_str1 < 0 ? 0 : y_str1;

		//Padding
		int loopr1 = ker1, loopc1 = ker1;

		//Upper
		if(row1 < pad1){
			loopr1 = ker1 - pad1;
		}
		//Bottom
		if(row1 >= out_fm1 - pad1){
			loopr1 = in_fm1 - x_str1/in_fm1;
		}
		//Left
		if(col1 < pad1){
			loopc1 = ker1 - pad1;
		}
		//Right
		if(col1 >= out_fm1 - pad1){
			loopc1 = in_fm1 -  y_str1;
		}

		float max1[5] = {0.0};
		for(int i = 0; i < model_num1; i++){
			for(int j = 0; j < loopr1; j++){
				for(int k = 0; k < loopc1; k++){
					if(max1[i] < (in_nu1[num_out*in_fm1*in_fm1 + j*in_fm1 + k + x_str1 + y_str1]))
						max1[i] = in_nu1[num_out*in_fm1*in_fm1 + j*in_fm1 + k + x_str1 + y_str1];
				}
			}
			out_nu1[out_position1] = max1[i];	
		}
	}
	if((blockIdx.x < model2_bidx) && (blockIdx.y < model2_bidyz) && (blockIdx.z < model2_bidyz) && (threadIdx.x < model2_tidxy) && (threadIdx.y < model2_tidxy))
	{
		int row2 = ((model2_tidxy*row_out_block)+(row_out_thread));
		int col2 = ((model2_tidxy*col_out_block)+(col_out_thread));
	
		int out_position2 = (out_fm2*out_fm2*num_out)
						+ (out_fm2*row2)
						+ col2;
		//Stride
		int x_str2 = 0, y_str2 = 0;
		x_str2 = (row2*str2-pad2)*in_fm2;
		x_str2 = x_str2 < 0 ? 0 : x_str2;
		y_str2 = col2*str2-pad2;
		y_str2 = y_str2 < 0 ? 0 : y_str2;

		//Padding
		int loopr2 = ker2, loopc2 = ker2;

		//Upper
		if(row2 < pad2){
			loopr2 = ker2 - pad2;
		}
		//Bottom
		if(row2 >= out_fm2 - pad2){
			loopr2 = in_fm2 - x_str2/in_fm2;
		}
		//Left
		if(col2 < pad2){
			loopc2 = ker2 - pad2;
		}
		//Right
		if(col2 >= out_fm2 - pad2){
			loopc2 = in_fm2 -  y_str2;
		}

		float max2[5] = {0.0};
		for(int i = 0; i < model_num2; i++){
			for(int j = 0; j < loopr2; j++){
				for(int k = 0; k < loopc2; k++){
					if(max2[i] < (in_nu2[num_out*in_fm2*in_fm2 + j*in_fm2 + k + x_str2 + y_str2]))
						max2[i] = in_nu2[num_out*in_fm2*in_fm2 + j*in_fm2 + k + x_str2 + y_str2];
				}
			}
			out_nu2[out_position2] = max2[i];	
		}
	}
}

__global__ void fused_two_conv(float *a_bias,float *a_in_w,float *r_in_w,
float *a_in_nu,float *r_in_nu,
float *a_out_nu,float *r_out_nu,
int alex_num,int res_num,
int a_in_fm,int a_out_fm,int a_str,int a_pad,int a_ker,int a_ker_channel,bool a_relu,
int r_in_fm,int r_out_fm,int r_str,int r_pad,int r_ker,int r_ker_channel,bool r_relu,
int model1_bidx,int model1_bidyz,int model1_tidxy,
int model2_bidx,int model2_bidyz,int model2_tidxy)
{
	// Alexnet + Resnet18
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

	if((blockIdx.x < model1_bidx) && (blockIdx.y < model1_bidyz) && (blockIdx.z < model1_bidyz) && (threadIdx.x < model1_tidxy) && (threadIdx.y < model1_tidxy))
	{
		int a_row = ((model1_tidxy*row_out_block)+(row_out_thread));
		int a_col = ((model1_tidxy*col_out_block)+(col_out_thread));
	
		int a_out_position = (a_out_fm*a_out_fm*num_out)
						+ (a_out_fm*a_row)
						+ a_col;

		//Stride
		int a_x_str = 0, a_y_str = 0;
		a_x_str = (a_row*a_str-a_pad)*a_in_fm;
		a_x_str = a_x_str < 0 ? 0 : a_x_str;
		a_y_str = a_col*a_str-a_pad;
		a_y_str = a_y_str < 0 ? 0 : a_y_str;

		//Padding
		int a_x_pad = 0, a_y_pad = 0;
		int a_loopr = a_ker, a_loopc = a_ker;

		//Upper
		if(a_row*a_str < a_pad){
			a_x_pad = a_pad - a_row*a_str;
			a_loopr = a_ker - a_x_pad;
		}
		//Bottom
		if(a_row >= a_out_fm - a_pad){
			a_loopr = a_in_fm - a_x_str/a_in_fm;
		}
		//Left
		if(a_col*a_str < a_pad){
			a_y_pad = a_pad - a_col*a_str;
			a_loopc = a_ker - a_y_pad;
		}
		//Right
		if(a_col >= a_out_fm - a_pad){
			a_loopc = a_in_fm -  a_y_str;
		}
		
		float a_product[5] = {0.0};
		for(int i = 0; i < alex_num; i++){
			for(int j = 0; j < a_ker_channel; j++){
				for(int k = 0; k < a_loopr; k++){
					for(int l = 0; l < a_loopc; l++){
						a_product[i] += a_in_nu[a_in_fm*a_in_fm*j + a_in_fm*k + l + a_x_str + a_y_str] 
								*a_in_w[num_out*a_ker_channel*a_ker*a_ker + j*a_ker*a_ker + k*a_ker + l + a_x_pad*a_ker + a_y_pad];
					}
				}
			}
			if(a_loopc > 0 && a_loopr > 0){
				a_product[i] += a_bias[num_out];

				//ReLU
				if(a_relu == true){
					if(a_product[i] < 0)
						a_product[i] = 0;
				}
				a_out_nu[a_out_position] = a_product[i];
			}
		}
	}

	if((blockIdx.x < model2_bidx) && (blockIdx.y < model2_bidyz) && (blockIdx.z < model2_bidyz) && (threadIdx.x < model2_tidxy) && (threadIdx.y < model2_tidxy))
	{
		int r_row = ((model2_tidxy*row_out_block)+(row_out_thread));
		int r_col = ((model2_tidxy*col_out_block)+(col_out_thread));

		int r_out_position = (r_out_fm*r_out_fm*num_out)
						+ (r_out_fm*r_row)
						+ r_col;
		//Stride
		int r_x_str = 0, r_y_str = 0;
		r_x_str = (r_row*r_str-r_pad)*r_in_fm;
		r_x_str = r_x_str < 0 ? 0 : r_x_str;
		r_y_str = r_col*r_str-r_pad;
		r_y_str = r_y_str < 0 ? 0 : r_y_str;

		//Padding
		int r_x_pad = 0, r_y_pad = 0;
		int r_loopr = r_ker, r_loopc = r_ker;

		//Upper
		if(r_row*r_str < r_pad){
			r_x_pad = r_pad - r_row*r_str;
			r_loopr = r_ker - r_x_pad;
		}
		//Bottom
		if(r_row >= r_out_fm - r_pad){
			r_loopr = r_in_fm - r_x_str/r_in_fm;
		}
		//Left
		if(r_col*r_str < r_pad){
			r_y_pad = r_pad - r_col*r_str;
			r_loopc = r_ker - r_y_pad;
		}
		//Right
		if(r_col >= r_out_fm - r_pad){
			r_loopc = r_in_fm - r_y_str;
		}

		float r_product[5] = {0.0};
		for(int i = 0; i < res_num; i++){
			for(int j = 0; j < r_ker_channel; j++){
				for(int k = 0; k < r_loopr; k++){
					for(int l = 0; l < r_loopc; l++){
						r_product[i] += r_in_nu[r_in_fm*r_in_fm*j + r_in_fm*k + l + r_x_str + r_y_str] 
								*r_in_w[num_out*r_ker_channel*r_ker*r_ker + j*r_ker*r_ker + k*r_ker + l + r_x_pad*r_ker + r_y_pad];
					}
				}
			}
			if(r_loopc > 0 && r_loopr > 0){
				//ReLU
				if(r_relu == true){
					if(r_product[i] < 0)
						r_product[i] = 0;
				}
				r_out_nu[r_out_position] = r_product[i];
			}
		}
	}
}

__global__ void fused_bn_max1(float *in_nu1,float *in_nu2,
float *out_nu1,float *out_nu2,
int model_num1,int model_num2,
float *mean,float *var,float *gamma,float *beta,int out_fm1,bool relu,
int in_fm2,int out_fm2,int str2,int pad2,int ker2,
int model1_bidx,int model1_bidyz,int model1_tidxy,
int model2_bidx,int model2_bidyz,int model2_tidxy)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

	if((blockIdx.x < model1_bidx) && (blockIdx.y < model1_bidyz) && (blockIdx.z < model1_bidyz) && (threadIdx.x < model1_tidxy) && (threadIdx.y < model1_tidxy))
	{
		int row1 = ((model1_tidxy*row_out_block)+(row_out_thread));
    	int col1 = ((model1_tidxy*col_out_block)+(col_out_thread));

		int out_position1 = (out_fm1*out_fm1*num_out)
						+ (out_fm1*row1)
						+ col1;

		float product1[5] = {0.0};
		for(int i = 0; i < model_num1; i++){
			product1[i] = ((in_nu1[out_position1] - mean[num_out])/(sqrt(var[num_out] + 1e-5)))*gamma[num_out] + beta[num_out];
			//ReLU
			if(relu == true){
				{
					if(product1[i] < 0)
						product1[i] = 0;
				}
   			}
			out_nu1[out_position1] = product1[i];
		}
	}

	if((blockIdx.x < model2_bidx) && (blockIdx.y < model2_bidyz) && (blockIdx.z < model2_bidyz) && (threadIdx.x < model2_tidxy) && (threadIdx.y < model2_tidxy)){
		

		int row2 = ((model2_tidxy*row_out_block)+(row_out_thread));
    	int col2 = ((model2_tidxy*col_out_block)+(col_out_thread));
	
		int out_position2 = (out_fm2*out_fm2*num_out)
						+ (out_fm2*row2)
						+ col2;
		
		//Stride
		int x_str = 0, y_str = 0;
		x_str = (row2*str2-pad2)*in_fm2;
		x_str = x_str < 0 ? 0 : x_str;
		y_str = col2*str2-pad2;
		y_str = y_str < 0 ? 0 : y_str;

		//Padding
		int loopr = ker2, loopc = ker2;

		//Upper
		if(row2 < pad2){
			loopr = ker2 - pad2;
		}
		//Bottom
		if(row2 >= out_fm2 - pad2){
			loopr = in_fm2 - x_str/in_fm2;
		}
		//Left
		if(col2 < pad2){
			loopc = ker2 - pad2;
		}
		//Right
		if(col2 >= out_fm2 - pad2){
			loopc = in_fm2 -  y_str;
		}

		float max[5] = {0.0};
		for(int i = 0; i < model_num2; i++){
			for(int j = 0; j < loopr; j++){
				for(int k = 0; k < loopc; k++){
					if(max[i] < (in_nu2[num_out*in_fm2*in_fm2 + j*in_fm2 + k + x_str + y_str]))
						max[i] = in_nu2[num_out*in_fm2*in_fm2 + j*in_fm2 + k + x_str + y_str];
				}				
			}
			out_nu2[out_position2] = max[i];
		}
	}
}

__global__ void fused_two_fc1(float *a_bias,float *r_bias,float *a_in_w,float *r_in_w,
float *a_in_nu,float *r_in_nu,
float *a_out_nu,float *r_out_nu,
int alex_num, int res_num,
int a_input, bool a_relu,
int r_input, bool r_relu)
{
	// Only Alexnet + Vgg16
	int num_out = blockIdx.x;

	int a_weight = num_out * a_input;
	float a_result[5] = {0.0};
	for(int i = 0; i < alex_num; i++){
		for(int j = 0; j < a_input; j++){
			a_result[i] += a_in_nu[j] * a_in_w[a_weight+j];
		}
		a_result[i] += a_bias[num_out];

		//ReLU
		if(a_relu == true){
			if(a_result[i] < 0)
				a_result[i] = 0;
		}

		a_out_nu[num_out] = a_result[i];
	}

	int r_weight = num_out * r_input;
	float r_result[5] = {0.0};
	for(int i = 0; i < res_num; i++){
		for(int j = 0; j < r_input; j++){
			r_result[i] += r_in_nu[j] * r_in_w[r_weight+j];
		}
		r_result[i] += r_bias[num_out];

		//ReLU
		if(r_relu == true){
			if(r_result[i] < 0)
				r_result[i] = 0;
		}

		r_out_nu[num_out] = r_result[i];
	}
}

//////////////////////////Single kernel//////////////////////////////////

__global__ void first_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,bool b,bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out)
					 + (out_fm*row)
					 + col;
	//Stride
    int x_str = 0, y_str = 0;
    x_str = 3*(row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = 3*(col*str-pad);
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
		loopr = in_fm - x_str/(3*in_fm);
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str/3;
	}

	float product[10] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < loopr; i++){
			for(int j = 0; j < loopc; j++){
				for(int k = 0; k < ker_channel; k++){
					product[l] += in_nu[i*in_fm*ker_channel + j*ker_channel + k + x_str + y_str] 
							*in_w[num_out*ker*ker*ker_channel + i*ker + j + k*ker*ker + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				if(product[l] < 0)
					product[l] = 0;
			}
			out_nu[out_position] = product[l];
		}
	}
}

/* Convolution */
__global__ void conv_jjb1(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,bool b,bool relu)
{
	int num_out = blockIdx.x + 64;
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

	float product[10] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < ker_channel; i++){
			for(int j = 0; j < loopr; j++){
				for(int k = 0; k < loopc; k++){
					product[l] += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
							*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				if(product[l] < 0) 
					product[l] = 0;
			}
			out_nu[out_position] = product[l];
		}
	}
}

__global__ void conv_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
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

	float product[10] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < ker_channel; i++){
			for(int j = 0; j < loopr; j++){
				for(int k = 0; k < loopc; k++){
					product[l] += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
							*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				if(product[l] < 0) 
					product[l] = 0;
			}
			out_nu[out_position] = product[l];
		}
	}
}

/* Local Response Normalization */
__global__ void norm_jjb(float *in_nu,float *out_nu,int model_num,
float alpha,float beta,int local_size,int out_fm)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    int input_position = (out_fm*row) + col;

    int nStart = 0, nEnd = 0;
    nStart=(num_out-2) > 1 ? (num_out-2) : 1 ;
    nEnd=(num_out+2) < gridDim.x ? (num_out+2) : gridDim.x ;

    float sum[10] = {0.0};
    float result[10] = {0.0};  
	for(int i = 0; i < model_num; i++){
		for(int j = (nStart-1); j < (nEnd-1); j++){
			sum[i] += pow((in_nu[j*out_fm*out_fm + input_position]),2);
		}
		result[i] = (in_nu[out_position]) / (pow( 1 + ((alpha/local_size) * sum[i]),beta));
		sum[i] = 0.0;
		out_nu[out_position] = result[i];
	}
}

/* Maxpooling */
__global__ void max_jjb(float *in_nu,float *out_nu,int model_num,
int in_fm,int out_fm,int str,int pad,int ker)
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
	int loopr = ker, loopc = ker;

	//Upper
	if(row < pad){
		loopr = ker - pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col < pad){
		loopc = ker - pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

    float max[10] = {0.0};
	for(int i = 0; i < model_num; i++){
		for(int j = 0; j < loopr; j++){
			for(int k = 0; k < loopc; k++){
				if(max[i] < (in_nu[num_out*in_fm*in_fm + j*in_fm + k + x_str + y_str]))
					max[i] = in_nu[num_out*in_fm*in_fm + j*in_fm + k + x_str + y_str];
			}
		}
    	out_nu[out_position] = max[i];
	}
}

/* Batch Normalization */
__global__ void batchnorm_jjb(float *in_nu,float *out_nu,int model_num,
float *mean,float *var,float *gamma,float *beta,int out_fm,bool relu)
{
    int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

	float product[10] = {0.0};
	for(int i = 0; i < model_num; i++){
		product[i] = ((in_nu[out_position] - mean[num_out])/(sqrt(var[num_out] + 1e-5)))*gamma[num_out] + beta[num_out];
		//ReLU
		if(relu == true){
			if(product[i] < 0)
				product[i] = 0;
		}
		out_nu[out_position] = product[i];
	}
}

/* Basic Block(in resnet) */
__global__ void basic_block_jjb(float *in_nu1,float *in_nu2,float *out_nu,int model_num,
int out_fm,bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

	float product[10] = {0.0};
	for(int i = 0; i < model_num; i++){
		product[i] = in_nu1[out_position] + in_nu2[out_position];
		//ReLU
		if(relu == true){
			if(product[i] < 0)
				product[i] = 0;
		}

		out_nu[out_position] = product[i];
	}
}

__global__ void globalavg_jjb(float *in_nu, float *out_nu,int model_num,int in_fm)
{
	int num_out = blockIdx.x;

	float sum[10] = {0.0};
	for(int k = 0; k < model_num; k++){
		for(int i = 0; i < in_fm; i++){
			for(int j = 0; j < in_fm; j++){
				sum[k] += in_nu[num_out*in_fm*in_fm + i*in_fm + j];
			}
		}
		out_nu[num_out] = sum[k]/(in_fm*in_fm);
	}
}

/* Fully Connected */
__global__ void fc_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
int input, bool relu)
{
    int num_out = blockIdx.x;
	int weight = num_out * input;

	float result[10] = {0.0};
	for(int i = 0; i < model_num; i++){
		for(int j = 0; j < input; j++){
			result[i] += in_nu[j] * in_w[weight+j];
		}
		result[i] += bias[num_out];
		//ReLU
		if(relu == true){
			if(result[i] < 0)
				result[i] = 0;
		}
		out_nu[num_out] = result[i];
	}
}

