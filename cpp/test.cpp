// test.cpp ---------------------------------------------------------------
#include "config.h"
#include "dataset.h"
#include "model.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <map>

namespace fs = std::filesystem;

/* ---------------------- simple helpers ---------------------- */
static double auc_roc(const std::vector<double>& score,
                      const std::vector<int>&    label)
{
    std::vector<size_t> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return score[a] > score[b]; });

    double tp = 0, fp = 0, tp_prev = 0, fp_prev = 0, auc = 0;
    const double P = std::count(label.begin(), label.end(), 1);
    const double N = label.size() - P;
    for (size_t k : idx) {
        if (label[k]) tp++; else fp++;
        if (fp != fp_prev)
            auc += (tp + tp_prev) * (fp - fp_prev) / 2.0;
        tp_prev = tp; fp_prev = fp;
    }
    return (P > 0 && N > 0) ? auc / (P * N) : 0.0;
}

/* ------------------------ main() ---------------------- */
int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: ./test_app <config.yaml>\n";
        return 1;
    }

    Config cfg = load_config(argv[1]);
    torch::Device dev(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Device: " << dev << "\n";

    auto test_ds = ADNIDataset(cfg, "test").map(torch::data::transforms::Stack<>());
    auto test_ld = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_ds),
        torch::data::DataLoaderOptions().batch_size(cfg.test_batch_sz).workers(cfg.workers));

    ClassifierCNN net(cfg); net->to(dev);
    const std::string model_path = cfg.file_name + ".pth";
    torch::load(net, model_path, dev);
    net->eval();
    std::cout << "Loaded model: " << model_path << "\n";

    int64_t correct = 0, total = 0;
    const int C = cfg.num_classes;
    std::vector<int64_t> all_pred, all_label;
    std::vector<std::vector<double>> all_prob(C);

    std::vector<std::vector<double>> rowwise_probs;  // NEW: stores full row-wise probabilities

    torch::NoGradGuard nograd;
    for (auto& b : *test_ld) {
        auto x = b.data.to(dev), y = b.target.to(dev);
        auto out = net->forward(x);
        auto prob = torch::softmax(out, 1);

        auto pred = out.argmax(1);
        correct += pred.eq(y).sum().item<int64_t>();
        total += y.size(0);

        auto pred_cpu = pred.cpu();
        auto label_cpu = y.cpu();
        auto prob_cpu = prob.cpu();

        for (int i = 0; i < y.size(0); ++i) {
            all_pred.push_back(pred_cpu[i].item<int64_t>());
            all_label.push_back(label_cpu[i].item<int64_t>());

            std::vector<double> probs_this_row;
            for (int c = 0; c < C; ++c) {
                double p = prob_cpu[i][c].item<double>();
                all_prob[c].push_back(p);
                probs_this_row.push_back(p);
            }
            rowwise_probs.push_back(probs_this_row);  // NEW
        }
    }

    const double acc = 100.0 * correct / std::max<int64_t>(1, total);

    std::vector<double> recall(C, 0.0), denom(C, 0.0);
    for (size_t i = 0; i < all_label.size(); ++i) {
        denom[all_label[i]] += 1.0;
        if (all_pred[i] == all_label[i])
            recall[all_label[i]] += 1.0;
    }
    for (int c = 0; c < C; ++c)
        recall[c] = denom[c] ? recall[c] / denom[c] : 0.0;
    const double bal_acc = 100.0 * std::accumulate(recall.begin(), recall.end(), 0.0) / C;

    std::vector<double> class_auc(C, 0.0);
    for (int c = 0; c < C; ++c) {
        std::vector<int> bin_label(all_label.size());
        std::transform(all_label.begin(), all_label.end(), bin_label.begin(),
                       [&](int64_t l) { return l == c ? 1 : 0; });
        class_auc[c] = auc_roc(all_prob[c], bin_label);
    }
    const double macro_auc = std::accumulate(class_auc.begin(), class_auc.end(), 0.0) / C;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Test Accuracy (simple) : " << acc << " %\n";
    std::cout << "Balanced Accuracy      : " << bal_acc << " %\n";
    std::cout << std::setprecision(4)
              << "Macro-AUC              : " << macro_auc << "\n";
    std::cout << "Per-class AUC          :";
    for (double a : class_auc) std::cout << " " << std::setprecision(4) << a;
    std::cout << "\n";

    /* ------------------------- save predictions ---------------------------- */
    const std::string tsv_in = cfg.test_tsv;
    const std::string tsv_out = "test_predictions.tsv";
    std::ifstream fin(tsv_in);
    std::ofstream fout(tsv_out);
    if (!fin) {
        std::cerr << "WARNING: cannot open " << tsv_in
                  << "; predictions TSV not written.\n";
        return 0;
    }

    std::string header;
    std::getline(fin, header);
    fout << header;

    // Output headers for class probabilities
    std::map<int, std::string> map_classes =
        (C == 3) ? std::map<int, std::string>{{0, "CN"}, {1, "LMCI"}, {2, "AD"}}
                 : std::map<int, std::string>{{0, "CN"}, {1, "AD"}};

    for (int i = 0; i < C; ++i)
        fout << "\tprob_" << map_classes[i];

    fout << "\ttrue_label\tpredicted_label\n";

    size_t row = 0;
    for (std::string line; std::getline(fin, line); ++row) {
        if (row >= all_label.size()) break;
        fout << line;
        for (int c = 0; c < C; ++c)
            fout << "\t" << std::fixed << std::setprecision(6) << rowwise_probs[row][c];
        fout << "\t" << map_classes[all_label[row]]
             << "\t" << map_classes[all_pred[row]] << "\n";
    }

    std::cout << "Predictions saved to " << tsv_out << "\n";
}
