#ifndef INCLUDE_LOAD_H
#define INCLUDE_LOAD_H

#include "opencv2/core.hpp"
#include "torch/torch.h"
#include "torch/script.h"
#include <string>
// #include "opencv2/opencv.hpp"

torch::Tensor loadImage(std::string& imagePath, cv::Size imageSize);
void normalize(torch::Tensor& inputTensor);
void testTensorAdd();

#endif