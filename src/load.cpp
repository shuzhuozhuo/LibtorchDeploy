
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
// #include "opencv2/opencv.hpp"
// #include "opencv2/core.hpp"

#include "torch/torch.h"
#include <string>
#include "../include/load.h"


#include <iostream>
#include <memory>
#include <stdexcept>

// struct MyException : public std::exception
// {
//   const char * what () const throw ()
//   {
//     return "C++ Exception";
//   }
// };

void normalize(torch::Tensor& inputTensor){
    
    float stdand[] = {0.225, 0.224, 0.229};
    torch::Tensor imageNetStd = torch::from_blob(stdand, {3}, torch::kFloat);

    float mean[] = {0.406, 0.456, 0.485};
    torch::Tensor imageNetMean = torch::from_blob(mean, {3}, torch::kFloat);

    for(int i = 0; i < 3; i ++){
        inputTensor.slice(0).slice(i) = inputTensor.slice(0).slice(i).sub(imageNetMean.slice(0, i, i + 1));
        inputTensor.slice(0).slice(i) = inputTensor.slice(0).slice(i).div(imageNetStd.slice(0, i, i+ 1));
    }
}

torch::Tensor loadImage(std::string& imagePath, cv::Size imageSize){

    cv::Mat image = cv::imread(imagePath, 1);
    if(image.empty()){
        throw std::invalid_argument("MyFunc argument too large.");
    }

    cv::resize(image, image, imageSize);

    torch::Tensor imageTensor;

    imageTensor = torch::from_blob(image.data, {imageSize.width, imageSize.height, 3}, torch::kByte);
    imageTensor = imageTensor.permute({2, 1, 0});
    imageTensor = torch::unsqueeze(imageTensor, 0);
    imageTensor = imageTensor.toType(torch::kFloat);
    normalize(imageTensor);
    return imageTensor;
}

void testTensorAdd(){
    double data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    torch::Tensor mytensor;
    mytensor = torch::from_blob(data, {2, 2, 2}, torch::kDouble);

    double stdand[] = {1, 1, 1};
    torch::Tensor imageNetStd = torch::from_blob(stdand, {3}, torch::kDouble);

    mytensor.slice(0, 0, 1) = mytensor.slice(0, 0, 1).sub(imageNetStd.slice(0, 0, 1));
    std::cout << mytensor << std::endl;
    std::cout << mytensor.slice(0, 0, 1) << std::endl;
    std::cout << imageNetStd.slice(0, 0, 1) << std::endl;
}