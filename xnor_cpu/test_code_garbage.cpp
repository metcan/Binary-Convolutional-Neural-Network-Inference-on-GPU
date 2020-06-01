	// //Multi core speed test
    // //This code is for single core vs multi core speed test. Also, it check variables to make sure the matrix order foes not change  
	// auto start = std::chrono::high_resolution_clock::now();
    // auto stop = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> multi_core(stop - start);
    // std::cout << multi_core.count() << std::endl;
	// auto start_1 = std::chrono::high_resolution_clock::now();
	// for (k=0; input_tensor.channel>k; k++)
	// {
	// 	intMat2BinaryMat<int>(input_tensor[k], binary_tensor_1[k], kernel_size);
	// }
    // auto stop_1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> single_core_1(stop_1 - start_1);

	// for (k=0; binary_tensor.size()>k; k++)
	// {
	// 	for(int j=0; binary_tensor[0].size()>j; j++)
	// 	{
	// 		for(int i=0; binary_tensor[0][0].size()>i; i++)
	// 		{
	// 			if (binary_tensor[k][j][i] == binary_tensor_1[k][j][i])
	// 			{
	// 				int as = 5;
	// 			}
	// 			else
	// 			{
	// 				std::cout<<"Error "<< binary_tensor[k][j][i] << " is not same with " << binary_tensor_1[k][j][i] <<std::endl;
	// 				break;
	// 			}
				
	// 		}
	// 	}
	//}


// Delete this function for release
// void xnor_debug_and_multithread()
// {
// 	Tensor<int> input_tensor(64, 64, 8);
// 	Weight<int> weight(3, 3, input_tensor.channel, 128);
// 	std::vector<float> scalar(weight.channel_out);
// 	std::pair<int, int> kernel_size(weight.row, weight.col);
// 	// Random initilizate the values
// 	for(int k=0; input_tensor.channel>k; k++)
// 	{
// 		for(int j=0; input_tensor.col>j; j++)
// 		{
// 			for(int i=0; input_tensor.row>i; i++)
// 			{
// 				input_tensor[k][j][i] = (std::rand()%1000 - 500);
// 				if (input_tensor[k][j][i] >= 0)
// 				{
// 					input_tensor[k][j][i] = 1;
// 				}
// 				else
// 				{
// 					input_tensor[k][j][i] = -1;
// 				}
				
// 			}
// 		}
// 	}
// 	for (int m=0; weight.channel_out>m; m++)
// 	{
// 		scalar[m] = static_cast<float>(rand()) / static_cast<float> (RAND_MAX);
// 		for(int k=0; weight.channel_in>k; k++)
// 		{
			
// 			for(int j=0; weight.col>j; j++)
// 			{
// 				for(int i=0; weight.row>i; i++)
// 				{
// 					weight[m][k][j][i] = (std::rand()%1000 - 500);
// 					if (weight[m][k][j][i] >= 0)
// 					{
// 						weight[m][k][j][i] = 1;
// 					}
// 					else
// 					{
// 						weight[m][k][j][i] = -1;
// 					}
					
// 				}
// 			}
// 		}
// 	}
// 	// Calculate hash map
// 	auto hash_map =  generate_hash_map(kernel_size);

// 	// Allocate binary tensor memory
// 	std::vector<Matrix<unsigned long>> binary_tensor_;
// 	for (int k=0; input_tensor.channel>k; k++)
// 	{
// 		binary_tensor_.push_back(BinaryMatMemoryAllocation<unsigned long>(std::make_pair(input_tensor[k].row, input_tensor[k].col), kernel_size) );
// 	}
// 	// Convert Int Matrix to Binary Mat
// 	{
// 		int k = 0;
// 		#pragma omp parallel private(k) shared(input_tensor, binary_tensor_)
// 		{
// 			#pragma omp for schedule(dynamic,50) collapse(1)
// 			for (k=0; input_tensor.channel > k ; k++)
// 				{
// 					intMat2BinaryMat<int>(input_tensor[k], binary_tensor_[k], kernel_size);
// 				}
// 		}
// 	}
// 	Tensor<unsigned long> binary_tensor(binary_tensor_[0].col, binary_tensor_[0].row, binary_tensor_.size()); 
// 	// Allocate binary weight memory
// 	std::vector<Tensor<unsigned long>> binary_weight_;
// 	for (int k=0; weight.channel_out>k; k++)
// 	{
// 		std::vector<Matrix<unsigned long>> binary_buffer_tensor;
// 		for(int j=0; weight.channel_in>j; j++)
// 		{
// 			binary_buffer_tensor.push_back(BinaryMatMemoryAllocation<unsigned long>
// 			(std::make_pair(weight.row, weight.col), kernel_size) );
// 		}
// 		Tensor<unsigned long> weight_tensor(weight.row, weight.col, weight.channel_in, binary_buffer_tensor);
// 		binary_weight_.push_back(weight_tensor); 
// 	}
// 	// Convert weights to binary
// 	Weight<unsigned long> binary_weight(1, 1, binary_weight_[0].size(), binary_weight_.size(), binary_weight_);
// 	for (int k=0; weight.channel_out>k; k++)
// 	{
// 		for(int j=0; weight.channel_in>j; j++)
// 		{
// 			intMat2BinaryMat(weight[k][j], binary_weight[k][j], kernel_size);
// 		}

// 	}
// 	// Generate K matrix
// 	auto A = tensorChannelSum<float>(input_tensor);
// 	Matrix<float> K(A.col - weight.col + 1, A.row - weight.row + 1);
// 	conv2D<float>(A, K, kernel_size);
// 	// Binary Convolution
// 		Tensor<unsigned long> output_tensor((input_tensor.col - weight.col + 1), (input_tensor.row - weight.row + 1), weight.channel_out);
	
// 	{
// 		int in;
// 		int out;
// 		#pragma omp parallel private(in, out) shared(binary_weight, output_tensor, binary_tensor)
// 		{
// 			#pragma omp for  collapse(2) schedule(dynamic, 50)
// 			for (out=0;weight.channel_out>out; out++)
// 				{
// 					for (in=0; weight.channel_in > in ; in++)
// 					{
// 						binaryConv2D(binary_tensor[in], output_tensor[out],
// 						binary_weight[out][in][0][0], std::make_pair(weight.col, weight.row) , std::make_pair(input_tensor.col, input_tensor.row));
// 					}
// 				}

// 		}
// 	}
// 	// Convert Binary convolution result to integer
// 	{
// 		int k;
// 		#pragma omp parallel private(k) shared(output_tensor, hash_map)
// 		{
// 			#pragma omp for collapse(1) schedule(dynamic, 50)
// 			for (k=0; output_tensor.channel>k; k++)
// 			{
// 				binaryMat2IntMat(output_tensor[k], hash_map);
// 			}
// 		}
// 	}
// 	// Multiplication with K and a scalar 
// 	Tensor<float> result_tensor(output_tensor.col, output_tensor.row, output_tensor.channel);
// 	for (int k=0; output_tensor.channel>k; k++)
// 	{
// 		for (int j=0; output_tensor.row>j; j++)
// 		{
// 			for(int i=0; output_tensor.row>i; i++)
// 			{
// 				result_tensor[k][j][i] = static_cast<float>(output_tensor[k][j][i]);
// 			}
// 		}
// 	}
// 	result_tensor *= K;
// 	result_tensor *= scalar;
// 	// End of BinaryConv
	
// 	return 0;
// }






