#include<iostream>
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "torch/torch.h"
#include "torch/script.h"

#include "./include/load.h"

int main(int argc, const char* argv[]){

    std::cout << "Hello World" << std::endl;
    cv::Size image_size(224, 224);
    printf("%d\n", image_size.height);
    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path, 1);
    torch::Tensor mytensor = loadImage(image_path, image_size);

    std::string model_path = "/home/shu/Projects/LearnPytorch/traced_resnet_model.pt";
    torch::jit::script::Module model;
    model = torch::jit::load(model_path);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(mytensor);
    at::Tensor output = model.forward(inputs).toTensor();
    at::Tensor max_index = torch::argmax(output);
    at::Tensor max_value = torch::max(output);
    // std::cout << "output is : " << output << std::endl;
    std::cout << "output index is : " << max_index << std::endl;
    std::cout << "output value is : " << max_value << std::endl;
    std::cout << "output size is : " << output.size(1) << std::endl;

    // testTensorAdd();
    return 1;
}