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
	}