// train.cpp
#include <torch/torch.h>
#include <iostream>
#include "model.h"
#include "dataset.h"

int main() {
    torch::manual_seed(123);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);

    // paths & hyperparams
    const std::string scans_dir = "/project/home/p200895/ADNI_processed"; // ../ADNI_processed
    const std::string train_tsv = "./participants_Train50_updated.tsv";
    const std::string  val_tsv  = "./participants_Val50_updated.tsv";
    const int    num_classes   = 3;
    const int    batch_size    = 4; // 16
    const size_t num_epochs    = 200; // 100
    const double lr            = 1e-3;

    // datasets + loaders (parallelized I/O via workers=4)
    auto train_ds = ADNIDataset(scans_dir, train_tsv, num_classes)
                .map(torch::data::transforms::Stack<>());
    auto val_ds   = ADNIDataset(scans_dir,  val_tsv, num_classes)
                .map(torch::data::transforms::Stack<>());

    // -------- DataLoaders ---------
    auto train_loader = torch::data::make_data_loader<
            torch::data::samplers::RandomSampler>(
            std::move(train_ds),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    auto val_loader   = torch::data::make_data_loader<
            torch::data::samplers::SequentialSampler>(
            std::move(val_ds),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));


    // model, loss, optimizer
    Simple3DCNN model(/*in_channels=*/1, num_classes);
    model->to(device);
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::Adam optimizer(model->parameters(), lr);

    double best_val_loss = float("+inf");
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        // -- TRAIN --
        model->train();
        double train_loss = 0, train_correct = 0, train_total = 0;
        for (auto& batch : *train_loader) {
            auto data   = batch.data.to(device);
            auto target = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss   = criterion(output, target);
            loss.backward();
            optimizer.step();

            train_loss   += loss.item<double>() * data.size(0);
            train_correct+= output.argmax(1).eq(target).sum().item<int64_t>();
            train_total  += data.size(0);
        }
        train_loss /= train_total;
        double train_acc = train_correct / train_total * 100.0;

        // -- VALIDATE --
        model->eval();
        torch::NoGradGuard ng;
        double val_loss = 0, val_correct = 0, val_total = 0;
        for (auto& batch : *val_loader) {
            auto data   = batch.data.to(device);
            auto target = batch.target.to(device);
            auto output = model->forward(data);
            auto loss   = criterion(output, target);

            val_loss    += loss.item<double>() * data.size(0);
            val_correct += output.argmax(1).eq(target).sum().item<int64_t>();
            val_total   += data.size(0);
        }
        val_loss /= val_total;
        double val_acc = val_correct / val_total * 100.0;

        // log & checkpoint
        std::cout << "Epoch ["<< epoch <<"/"<< num_epochs <<"] "
                  << "Train: loss="<<train_loss<<" acc="<<train_acc<<"%  "
                  << "Val: loss="<<val_loss<<" acc="<<val_acc<<"%\n";

        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            torch::save(model, "best_model.pt");
            std::cout<<"  ⇨ Saved new best model!\n";
        }
    }
    std::cout<<"Training complete. Best Val Loss: "<<best_val_loss<<"%\n";
    return 0;
}
