#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2\core.hpp>
#include <algorithm>

using namespace cv;

template <typename T>
void endswap(T *objp){
	unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
	std::reverse(memp, memp + sizeof(T));
}

void ReadLabels(std::string name_file, std::vector<unsigned char>& dst){
	std::ifstream f(name_file, std::ios::binary);
	if (!f){
		throw std::runtime_error("Error: file " + name_file + " - not found");
	}
	int MSB, size;
	f.read(reinterpret_cast<char*>(&MSB), sizeof(int));
	f.read(reinterpret_cast<char*>(&size), sizeof(int));
	endswap(&MSB);
	endswap(&size);
	char byte;
	for (int k = 0; k < size; ++k){
		f.read(&byte, sizeof(unsigned char));
		dst.push_back(byte);
	}
	f.close();
}

void ReadImages(std::string name_file, std::vector<Mat>& dst, int ConvertType = -1){
	std::ifstream f(name_file, std::ios::binary);
	if (!f){
		throw std::runtime_error("Error: file " + name_file + " - not found");
	}
	int MSB, size, rows, cols;
	f.read(reinterpret_cast<char*>(&MSB), sizeof(int));
	f.read(reinterpret_cast<char*>(&size), sizeof(int));
	f.read(reinterpret_cast<char*>(&rows), sizeof(int));
	f.read(reinterpret_cast<char*>(&cols), sizeof(int));
	endswap(&MSB);
	endswap(&size);
	endswap(&rows);
	endswap(&cols);
	char byte;
	for (int k = 0; k < size; ++k){
		Mat tmp(rows, cols, CV_8UC1);
		for (int i = 0; i < rows; ++i){
			for (int j = 0; j < cols; ++j){
				f.read(&byte, sizeof(unsigned char));
				tmp.at<uchar>(i, j) = byte;
			}
		}
		if (ConvertType != -1) tmp.convertTo(tmp, ConvertType);
		dst.push_back(tmp);
	}
	f.close();
}

