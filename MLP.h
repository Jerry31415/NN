#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <sstream>
#include "Function.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include <opencv2\core\core.hpp>

void Softmax(std::vector<double>& v);

class MLP{
	std::vector<double> r;
public:
	std::vector<int> LS;
	std::vector<double> w;

	MLP(){}
	MLP(const std::vector<int>& layers_size);
	int weightsNumber();
	int neuronNumber();

	void get(const std::vector<double>& arg, std::vector<double>& dst);
	void get(const std::vector<double>& arg, std::vector<double>& dst, const std::vector<double>& W);

	void getA(const std::vector<double>& arg, std::vector<double>& dst, const Function& activation_func);
	void getA(const std::vector<double>& arg, std::vector<double>& dst, const std::vector<double>& W, const Function& activation_func);

	void getA_ReLU(const std::vector<double>& arg, std::vector<double>& dst);
	void getA_ReLU(const std::vector<double>& arg, std::vector<double>& dst, const std::vector<double>& W);
	
	void SaveImage(const std::string& name, cv::Size size = cv::Size(1000, 1000));
};


/* TEST SUMMATORS
in: 1 2 -1 3 -1

L1 sum: 2 -4 12
L2 sum: 46 20
L3 sum: 98
L1:
1 -1 3 2 0 0
2 1 3 -4 3 0
6 2 -2 1 3 0
L2:
4 -1 2 0
5 1 2 0
L3:
3 0 -2 0

MLP mlp({ 5, 3, 2, 1 });
mlp.w = { 1, -1, 3, 2, 0, 0, 2, 1, 3, -4, 3, 0, 6, 2, -2, 1, 3, 0, 4, -1, 2, 0, 5, 1, 2, 0, 3, -2, 0 };

*/