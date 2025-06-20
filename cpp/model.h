// model.h
#pragma once
#include "config.h"
#include <torch/torch.h>
#include <iostream>

struct ClassifierCNNImpl : torch::nn::Module {
    torch::nn::Sequential conv{nullptr};
    torch::nn::Linear     fc6{nullptr};
    torch::nn::Sequential classifier{nullptr};

    explicit ClassifierCNNImpl(const Config& cfg) {
        const int in_c = cfg.in_channels;
        const int exp  = cfg.expansion;
        const int nc   = cfg.num_classes;
        const int fdim = cfg.feature_dim;
        const int nhid = cfg.nhid;
        const int crop = cfg.crop_size;
        const bool inst_norm =
            (cfg.norm_type == "Instance" || cfg.norm_type == "instance");

        auto Norm3d = [&](int c) -> torch::nn::AnyModule {
            return inst_norm ? torch::nn::AnyModule(torch::nn::InstanceNorm3d(c))
                             : torch::nn::AnyModule(torch::nn::BatchNorm3d(c));
        };

        conv = torch::nn::Sequential(
            // Layer 1 --------------------------------------------------------
            // 1x1 convolution to expand channels
            torch::nn::Conv3d(torch::nn::Conv3dOptions(in_c, 4*exp, 1)),
            Norm3d(4*exp),
            torch::nn::ReLU(true),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(3).stride(2)),

            // Layer 2 --------------------------------------------------------
            // 3x3 dilated convolution with stride 1
            torch::nn::Conv3d(torch::nn::Conv3dOptions(4*exp, 32*exp, 3)
                              .dilation(2)),
            Norm3d(32*exp),
            torch::nn::ReLU(true),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(3).stride(2)),

            // Layer 3 --------------------------------------------------------
            // 5x5 dilated convolution with padding
            torch::nn::Conv3d(torch::nn::Conv3dOptions(32*exp, 64*exp, 5)
                              .padding(2).dilation(2)),
            Norm3d(64*exp),
            torch::nn::ReLU(true),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(3).stride(2)),

            // Layer 4 --------------------------------------------------------
            // 3x3 dilated convolution with padding
            torch::nn::Conv3d(torch::nn::Conv3dOptions(64*exp, 64*exp, 3)
                              .padding(1).dilation(2)),
            Norm3d(64*exp),
            torch::nn::ReLU(true),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(5).stride(2))
        );
        register_module("conv", conv);

        // Dynamically compute the flattened size after conv layers --------
        int64_t flat_dim;
        {
            torch::NoGradGuard _;
            auto dummy = torch::zeros({1, in_c, crop, crop, crop});
            flat_dim   = conv->forward(dummy).view({1, -1}).size(1);
        }
        std::cout << "[Model] flat_dim = " << flat_dim << '\n';

        // Fully connected layer before the classifier ----------------------
        fc6 = register_module("fc6", torch::nn::Linear(flat_dim, fdim));

        // Classifier: hidden + output layer --------------------------------
        classifier = register_module("classifier",
                     torch::nn::Sequential(torch::nn::Linear(fdim, nhid),
                                           torch::nn::Linear(nhid, nc)));

        // Initialize linear layers' weights and biases ---------------------
        for (auto& m : modules(/*include_self=*/false))
            if (auto* l = dynamic_cast<torch::nn::LinearImpl*>(m.get())) {
                torch::nn::init::normal_(l->weight, 0.0, 0.01);
                torch::nn::init::constant_(l->bias,   0.0);
            }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = x.view({x.size(0), -1});
        x = fc6->forward(x);
        return classifier->forward(x);
    }
};
TORCH_MODULE(ClassifierCNN);
