// model.h
#pragma once
#include <torch/torch.h>

struct Simple3DCNNImpl : torch::nn::Module {
    // layers
    torch::nn::Conv3d    conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm3d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Linear    fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout   dropout{nullptr};

    Simple3DCNNImpl(int in_channels = 1, int num_classes = 3) {
        // conv blocks
        conv1 = register_module("conv1",
        torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, 32, 3)
                        .stride(1).padding(1)));
        bn1   = register_module("bn1", torch::nn::BatchNorm3d(32));

        conv2 = register_module("conv2",
                torch::nn::Conv3d(torch::nn::Conv3dOptions(32, 64, 3)
                                .stride(1).padding(1)));
        bn2   = register_module("bn2", torch::nn::BatchNorm3d(64));

        conv3 = register_module("conv3",
                torch::nn::Conv3d(torch::nn::Conv3dOptions(64, 128, 3)
                                .stride(1).padding(1)));
        bn3   = register_module("bn3", torch::nn::BatchNorm3d(128));

        // fully connected
        fc1     = register_module("fc1",     torch::nn::Linear(128 * 15 * 18 * 15, 512));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
        fc2     = register_module("fc2",     torch::nn::Linear(512, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::max_pool3d(torch::relu(bn1(conv1(x))), /*kernel_size=*/2);
        x = torch::max_pool3d(torch::relu(bn2(conv2(x))), 2);
        x = torch::max_pool3d(torch::relu(bn3(conv3(x))), 2);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1(x));
        x = dropout(x);
        x = fc2(x);
        return x;
    }
};
TORCH_MODULE(Simple3DCNN);
