#include <iostream>
#include <vector>
#include <stdlib.h>
#include <assert.h>  
#include <math.h> 
#include <unordered_map>
#include <algorithm> 
#include <functional>
// Indexs are defined with x,y coordinate pairs 


constexpr std::pair<int, int> register_size(8, 8);
// TODO txt reader and saver
// TODO add hash table to convert convolution results
void countSetBits(int x, int &y) 
{ 

    while (x) { 
        y += x & 1; 
        x >>= 1; 
    } 
} 

template<typename T>
std::vector<std::vector<T>> tensor_channel_sum(std::vector<std::vector<std::vector<T> > > &input_x)
{
	std::vector<std::vector<T>> output_y(input_x[0].size(), std::vector<T>(input_x[0][0].size(), 0) );
	for(int k=0; input_x.size() > k; k++)
	{
		for(int j=0; input_x[0].size()>j; j++)
		{
			int sum = 0;
			for (int i=0; input_x[0][0].size()>i; i++)
			{
				sum += input_x[k][j][i];
			}
			output_y[k][j] = sum / input_x[0][0].size();
		}
	}
	return output_y;
}

template<typename T>
std::vector<std::vector<double>> conv2D(std::vector<std::vector<T>> &input_x, std::pair<int, int>kernel_size)
{
	std::vector<std::vector<double> > output_y;
	std::vector<std::vector<double>> scaled_x(input_x.size(), std::vector<double>(input_x[0].size(), 0));

	double k = 1 / (input_x.size() * input_x[0].size());
	
	for (int j=0; input_x.size()>j; j++)
	{
		for (int i=0; input_x[0].size()>i; i++)
		{
			scaled_x[j][i] = input_x[j][i] * k;
			for(int y=(1 - kernel_size.second)/2 ; (kernel_size.second + 1)/2 > y; y++)
			{
				if ((j+ y>0) && (j + y<input_x.size()))
					{
						break;
					} 
				for(int x=(1- kernel_size.first)/2 ; (kernel_size.first + 1)/2 > x; x++)
				{

					if (((i+ x)>0) && ((i + x)<input_x[0].size() ) )
					{
						break;
					}
					output_y[j+y][i+x] += scaled_x[j][i]; 
				}

			}
			
		}
	}
	return output_y;
}



void recursive_hash_map(std::vector<int> &key_vector, std::unordered_map<long int, int> &hash_map,
 int &iteration_count, int count_index, const std::pair<int, int>kernel_size)
{
	for (int i=0; i < iteration_count; i++)
	{
		key_vector[count_index] = i;
		if (count_index == key_vector.size() - 1)
		{
			int key_value = 0;
			int bit_count = 0;
			for(int j=0; j<key_vector.size(); j++)
			{
				key_value += key_vector[j] * std::pow(2, register_size.first * j);  // Shift values so that they can be equal to 
				
			}
			countSetBits(key_value, bit_count);
			hash_map[key_value] = 2* bit_count - ( kernel_size.first * kernel_size.second );
		} 
		else
			recursive_hash_map(key_vector, hash_map, iteration_count, (count_index + 1), kernel_size);
	}
}




std::unordered_map<long int, int> generate_hash_map(std::pair<int, int> &kernel_size)
{
	std::unordered_map<long int, int> hash_map;
	std::vector<int> key_vector(kernel_size.second);
	int iteration_count = std::pow(2, kernel_size.first);
	int count_index = 0;
	recursive_hash_map(key_vector, hash_map, iteration_count, count_index, kernel_size);
	return hash_map;
}

template<typename T>
void float2binary(const std::vector<T> input_x, const std::pair<int, int> input_index, 
 std::pair<int, int> output_location, long int &output_y, const std::pair<int ,int>register_size)
{	
	int sign = 0;
	long int pozitif = 1;
	long int negatif = 0;
	int count = output_location.second * 8  + output_location.first;

	assert(count < register_size.second * register_size.first); 

	for(int i=0; i<register_size.first; i++)
	{
		sign = (input_x[input_index.first + i] > 0) - (input_x[input_index.first + i] < 0);
		if (sign == 1)
		{
			output_y = pozitif<<count | output_y;
		}
		else if (sign == -1)
		{
			output_y = negatif<<count | output_y;
		}
		else
		{
			output_y = negatif<<count |output_y;
		}
		if ((input_index.first + i) >=  input_x.size())
		{
			break;
		}
		count++;
		
	}

}

template<typename T>
std::vector<std::vector<long int>> floatMat2BinaryMat(std::vector<std::vector<T>> &input_image, std::pair<int, int> &kernel_size)
{
	const std::pair<int, int> register_size(8, 8); 
	int size_x = ceil((input_image[0].size() - register_size.first) 
						/static_cast<double>(register_size.first + 1 - kernel_size.first) + 1);
	int size_y = ceil((input_image.size() - register_size.second ) 
						/static_cast<double>(register_size.second + 1 - kernel_size.second) + 1);
	if (size_x < 0)
		size_x = 1;
	if (size_y < 0)
		size_y = 1;

	std::vector<long int> row_x(size_x); 
	std::vector<std::vector<long int> > output_image(size_y, row_x);
	int index_x = 0;
	int index_y = 0;
	std::pair<int, int> input_index(0, 0);
	std::pair<int, int> output_location(0, 0);
	
	while(input_image.size() > input_index.second)
	{
		std::vector<T> input_row = input_image[input_index.second];
		int i = 0;
		input_index.first = 0;
		index_x = 0;
		
		while(input_row.size() > i)
		{
			i = input_index.first + register_size.first; 
			float2binary<T>(input_row, input_index, output_location, output_image[index_y][index_x], register_size);
			input_index.first = input_index.first + register_size.first + 1 - kernel_size.first;
			index_x++;
			
		}
		output_location.second++;
		input_index.second++;
		if(input_index.second >= input_image.size())
			{
				break;
			}
		if (output_location.second % register_size.second == 0)
		{
			output_location.second = 0;
			input_index.second = input_index.second + 1 - kernel_size.second;
			index_y++;
		}
	}
	return output_image;
}
template<typename T>
void binary2floatVec(const long int &input_x, const std::pair<int, int> &output_index,
				 std::vector<std::vector<T> > &output_y, const std::pair<int, int> &output_size)
{	
	constexpr std::pair<int, int> register_size(8, 8);
	const std::pair<int, int> input_index(output_index.first/register_size.first,
											 output_index.second/register_size.second);
	long int shifter = input_index.second * register_size.second + input_index.first;
	constexpr long int mask = 1;
	int sign = 0;
	int index_x = 0;
	std::pair<int, int> iteration(0, 0);
	 
	if ( (output_index.first + register_size.first) >= output_size.first)
	{
		iteration.first = register_size.first;
	}
	else
	{
		iteration.first = output_size.first - output_index.first;
	}
	
	if 	((output_index.second  + register_size.second) >= output_size.second)
	{
		iteration.first = register_size.second;
	}
	else
	{
		iteration.second = output_size.second - output_index.second;
	}
	
	for (int j=0; iteration.second > j; j++)
	{
		for (int i=0; iteration.first > i; i++)
		{
			sign = ((input_x>>shifter)&(mask));
			if (sign == 1)
			{
				output_y[output_index.second + j][ output_index.first + i] = static_cast<T>(sign);
			}
			else if (sign == 0)
			{
				output_y[output_index.second + j][output_index.first + i] = -1;
			}
			else
			{
				std::cout << "Wrong result" << std::endl;
			}
			shifter++;
		}
	}
} 


//Result converter
template<typename T>
std::vector<std::vector<T> > binaryMat2FloatMat(std::vector<std::vector<long int> > input_img, std::pair<int, int> output_size)
{
	const std::pair<int, int> register_size(8, 8);
	std::vector<std::vector<T> >  output_image(output_size.second, std::vector<T>(output_size.first, 0));
	std::pair<int, int> output_index(0, 0);
	for(int j=0; input_img.size()>j; j++)
	{
		
		for(int i=0; input_img[0].size()>i; i++)
		{
			binary2floatVec(input_img[j][i], output_index,
				 output_image, output_size);
			output_index.first += register_size.first;



		}
		output_index.second += register_size.second;

	}
	return output_image;
}

void cellConv2D(long int input_img, long int conv_kernel, const long int mask,
				std::pair<int, int> conv_iter, std::pair<int, int> output_index, std::pair<int, int> image_size,
				std::vector<std::vector<long int>> &output_img)
{
	constexpr std::pair<int, int> register_size(8, 8);
	const std::pair<int, int> input_index(0, 0);
	long int shifter = 0;
	
	int sign = 0;
	int index_x = 0;
	 
	 // iteration parameters for convolution kernel
	 // X axis calculation
	if ( output_index.first + register_size.first < image_size.first)
	{
		conv_iter.first = conv_iter.first;
	}
	else
	{
		conv_iter.first = conv_iter.first - (register_size.first + output_index.first - image_size.first);
	}
	// Y axis calculation
	if 	((output_index.second  + register_size.second) < image_size.second)
	{
		conv_iter.second = conv_iter.second;
	}
	else
	{
		conv_iter.second = conv_iter.second - (register_size.second + output_index.second - image_size.second);
	}
	unsigned long int shift = 0;
	for (int j=0; conv_iter.second > j; j++)
	{
		for (int i=0; conv_iter.first > i; i++)
		{
			// COnvolution operation here
			shifter = i + j * register_size.second;
			output_img[output_index.second + j][output_index.second + i] = (input_img | (conv_kernel>>shifter))&mask;
		}
	}

}



std::vector<std::vector<long int>> binaryConv2D(std::vector<std::vector<long int>> input_img,
										long int conv_kernel, std::pair<int, int> conv_size,
										std::pair<int, int> image_size)
{

	std::vector<std::vector<long int>> output_img((image_size.second - (conv_size.second - 1)),
										 std::vector<long int> (image_size.first - (conv_size.first - 1), 0));
	const std::pair<int, int> register_size(8, 8);
	long int mask = 0;
	mask = std::pow(2, conv_size.first) - 1;
	for (int j=1; conv_size.second > j ; j++)
	{
		mask = (mask<<register_size.second) | static_cast<long int>(std::pow(2, conv_size.first) - 1);
	}
	// mask = 1110000011100000111 = 2^3 -1 - 2^8 + 2^11 - 2^ 

	const int conv_per_row = register_size.first - (conv_size.first - 1);
	const int conv_per_column = register_size.second - (conv_size.second - 1);
	std::pair<int, int> conv_iter = std::make_pair(conv_per_row, conv_per_column);
	std::pair<int, int> output_index(0, 0); 
	for(int j=0; input_img.size()>j; j++)
	{
		output_index.first = 0;
		for(int i=0; input_img[0].size()>i; i++)
		{
			cellConv2D(input_img[j][i], conv_kernel, mask,
				 		conv_iter, output_index, image_size,
						output_img);
			output_index.first += conv_per_row;
		}
		output_index.second += conv_per_column;

	}
	return output_img;
}

void binaryMat2IntMat(std::vector<std::vector<long int>> &input_x, const std::unordered_map<long int, int> &hash_map)
{
	for(int j=0; input_x.size()>j; j++)
	{
		for (int i=0; input_x[j].size()>i; i++)
		{
			input_x[j][i] = hash_map.at(input_x[j][i]);
		}
	}
}
int main()
{
	std::pair<int, int> image_size(8, 8);
	std::vector<std::vector<float> > img(image_size.first, std::vector<float>(image_size.second, 1));
	std::pair<int, int>kernel_size(3, 3);
	std::vector<std::vector<float> > kernel(kernel_size.first, std::vector<float>(kernel_size.second, 1));

	auto hash_map =  generate_hash_map( kernel_size);


	auto binary_img = floatMat2BinaryMat(img, kernel_size);
	auto kernel_1 = floatMat2BinaryMat(kernel, kernel_size);
	long binary_kernel = kernel_1[0][0];
	auto result_mat = binaryConv2D(binary_img, binary_kernel, kernel_size, image_size);
	for(int j=0; result_mat.size()>j; j++)
	{
		for (int i=0; result_mat[j].size()>i; i++)
		{
			std::cout<<result_mat[j][i] << "  ";
		}
		std::cout<<std::endl;
	}
	binaryMat2IntMat(result_mat, hash_map);
	for(int j=0; result_mat.size()>j; j++)
	{
		for (int i=0; result_mat[j].size()>i; i++)
		{
			std::cout<<result_mat[j][i] << "  ";
		}
		std::cout<<std::endl;
	}
	std::tuple<int, int, int> tensorsize_xyz = std::make_tuple(100, 256, 256);
	std::vector<std::vector<std::vector<double>>> input_tensor(std::get<0>(tensorsize_xyz),
	 std::vector<std::vector<double>>(std::get<1>(tensorsize_xyz), 
	 std::vector<double>(std::get<2>(tensorsize_xyz))));
	
	for (int k = 0; std::get<0>(tensorsize_xyz) > k; k++)
	{
		for (int j=0; std::get<1>(tensorsize_xyz)>j; j++)
		{
			for (int i=0; std::get<2>(tensorsize_xyz)>i; i++)
			{
				input_tensor[k][j][i] = (std::rand()%1000 + 1);
			}
		}
	}
	
	auto matrix_K = tensor_channel_sum(input_tensor);
	return 0;
}






