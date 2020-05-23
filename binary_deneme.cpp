#include <iostream>
#include <vector>
#include <stdlib.h>
#include <assert.h>  
#include <math.h> 

template<typename T>
std::vector<unsigned long int> float2binary(std::vector<T> input_x)
{
	// Maximum input size
	size_t output_size = static_cast<size_t>(input_x.size()/64) + (input_x.size()%64==0?0:1);
	std::vector<unsigned long int> output_y(output_size);
	std::fill(output_y.begin(), output_y.end(), 0);
	int index_y = 0; // output index counter
	int count = 0;
	int sign = 0;
	unsigned long int pozitif = 1;
	unsigned long int negatif = 0;  
	for(int i=0; i<input_x.size(); i++)
	{
		sign = (input_x[i] > 0) - (input_x[i] < 0);
		if (sign == 1)
		{
			output_y[index_y] = pozitif<<count | output_y[index_y];
		}
		else if (sign == -1)
		{
			output_y[index_y] = negatif<<count | output_y[index_y];
		}
		else
			output_y[index_y] = pozitif<<count |output_y[index_y];
		if (i>62 & (i%63==0))
		{
			index_y++;
			count = 0;
		}
		else
			count++;
	}
	return output_y;
}

template<typename T>
std::vector<T> binary2float(std::vector<unsigned long int> input_x)
{
	// Input size
	std::vector<T> output_y(size_t(input_x.size() * 64));

	unsigned long int shifter = 0;
	unsigned long int mask = 1;
	int sign = 0;
	int index_x = 0;
	for(int i=0; i < output_y.size(); i++)
	{
		sign = ((input_x[index_x]>>shifter)&(mask));
		if (sign == 1)
		{
			output_y[i] = static_cast<T>(sign);
		}
		else if (sign == -1)
		{
			output_y[i] = 0;
		}
		else
		{
			std::cout << "Wrong result" << std::endl;
		}
		
		//std::cout<< (input_x[index_x]>>shifter) << "  " << shifter << "  " << index_x << std::endl;
		shifter = shifter + 1;
		if (shifter == 63)
		{
			shifter = 0;
			index_x += 1;
		}
	}
	return output_y;
}
int main()
{
	std::vector<float> x(65);
	for (int i=0; x.size() > i; i++)
	{
		x[i] = (rand() % 10) + 5;
	}
	auto y = float2binary<float>(x);
	std::cout<< y.size()<< std::endl;
	auto z = binary2float<unsigned long int>(y);
	for(int i=0; i<z.size(); i++)
	{
		std::cout<< z[i] << "  ";
	}
	return 0;
}