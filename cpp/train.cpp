// train.cpp
#include "config.h"
#include "dataset.h"
#include "model.h"
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

/* train/val loop --------------------------------------------------------- */
template <typename Loader>
std::pair<double,double> run_epoch(ClassifierCNN& net, Loader& loader,
                                   bool train_phase, torch::Device dev,
                                   torch::nn::CrossEntropyLoss& crit,
                                   torch::optim::Optimizer* opt,
                                   double grad_clip)
{
    (train_phase ? net->train() : net->eval());
    torch::AutoGradMode mode(train_phase);

    double   loss_sum = 0.0;
    int64_t  correct  = 0, total = 0;

    for (auto& b : loader) {
        auto x = b.data.to(dev), y = b.target.to(dev);
        if (!x.numel()) continue;

        if (train_phase) opt->zero_grad();

        auto out  = net->forward(x);
        auto loss = crit(out, y);

        if (train_phase) {
            loss.backward();
            if (grad_clip > 0)
                torch::nn::utils::clip_grad_norm_(net->parameters(), grad_clip);
            opt->step();
        }

        loss_sum += loss.template item<double>() * x.size(0);
        correct  += out.argmax(1).eq(y).sum().template item<int64_t>();
        total    += x.size(0);
    }
    return {loss_sum / std::max<int64_t>(1,total),
            total ? 100.0 * double(correct) / total : 0.0};
}

/* main -------------------------------------------------------------------- */
int main(int argc,char* argv[]) {
    if (argc!=2){ std::cerr<<"usage: ./train_app <config.yaml>\n"; return 1; }

    Config cfg = load_config(argv[1]);
    torch::Device dev(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout<<"Device: "<<dev<<"\n";

    auto train_ds = ADNIDataset(cfg,"train").map(torch::data::transforms::Stack<>());
    auto val_ds   = ADNIDataset(cfg,"val"  ).map(torch::data::transforms::Stack<>());

    auto train_ld = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_ds),
        torch::data::DataLoaderOptions().batch_size(cfg.batch_sz).workers(cfg.workers));

    auto val_ld = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_ds),
        torch::data::DataLoaderOptions().batch_size(cfg.val_batch_sz).workers(cfg.workers));

    ClassifierCNN net(cfg); net->to(dev);

    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(cfg.lr));
    torch::nn::CrossEntropyLoss crit;

    fs::create_directories(fs::path(cfg.log_csv).parent_path());
    std::ofstream log(cfg.log_csv);
    log<<"epoch,train_loss,train_acc,val_loss,val_acc\n";

    double best = std::numeric_limits<double>::infinity();
    for (int e=1; e<=cfg.epochs; ++e){
        auto [tl,ta]=run_epoch(net,*train_ld,true ,dev,crit,&opt,cfg.grad_clip);
        auto [vl,va]=run_epoch(net,*val_ld  ,false,dev,crit,nullptr,0.0);

        std::cout<<"Epoch "<<e<<"/"<<cfg.epochs
                 <<"  TL "<<tl<<" ("<<ta<<"%)"
                 <<"  VL "<<vl<<" ("<<va<<"%)\n";

        log<<e<<','<<tl<<','<<ta<<','<<vl<<','<<va<<'\n'; log.flush();

        if (vl<best){
            best=vl;
            fs::create_directories(fs::path(cfg.file_name).parent_path());
            torch::save(net, cfg.file_name+".pt");
            std::cout<<"   ✔ best model saved\n";
        }
    }
}
