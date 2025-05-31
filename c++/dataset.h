// dataset.h
#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <nifti1_io.h>  // link with niftiio+znz

class ADNIDataset : public torch::data::datasets::Dataset<ADNIDataset> {
    std::vector<std::pair<std::string,int64_t>> samples_;

public:
    explicit ADNIDataset(const std::string& scans_dir,
                         const std::string& tsv_path,
                         int num_classes = 3)
    {
        // define valid labels
        std::vector<std::string> valid = (num_classes == 3)
            ? std::vector{"CN","LMCI","AD"}
            : std::vector{"CN","AD"};
        std::unordered_map<std::string,int64_t> label_map;
        for (size_t i = 0; i < valid.size(); ++i)
            label_map[valid[i]] = static_cast<int64_t>(i);

        // read TSV
        std::ifstream file(tsv_path);
        if (!file) throw std::runtime_error("Cannot open TSV: " + tsv_path);
        std::string line;
        std::getline(file, line);  // skip header

        namespace fs = std::filesystem;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string pid, sid, diag;
            std::getline(ss, pid, '\t');
            std::getline(ss, sid, '\t');
            std::getline(ss, diag, '\t');
            auto it = label_map.find(diag);
            if (it == label_map.end()) continue;

            fs::path p = fs::path(scans_dir) / "subjects" / pid / sid /
                         "t1"/"spm"/"segmentation"/"normalized_space";
            if (!fs::exists(p)) continue;

            for (auto& e : fs::directory_iterator(p)) {
                auto fn = e.path().filename().string();
                if (fn.find("Space_T1w") != std::string::npos) {
                    samples_.emplace_back(e.path().string(), it->second);
                    break;
                }
            }
        }
    }

    // override get() and size()
    torch::data::Example<> get(size_t idx) override {
        auto [path, lbl] = samples_.at(idx);

        // load NIfTI
        nifti_image* img = nifti_image_read(path.c_str(), /*read_data=*/1);
        if (!img) throw std::runtime_error("Failed NIfTI read: " + path);

        int X = img->nx, Y = img->ny, Z = img->nz;
        float* raw = static_cast<float*>(img->data);
        // note: NIfTI is X×Y×Z; we want Z×Y×X
        auto tensor = torch::from_blob(raw, {Z,Y,X}, torch::kFloat32).clone();
        nifti_image_free(img);

        // add channel, normalize
        tensor = tensor.unsqueeze(0);
        auto mn = tensor.min().item<float>();
        auto mx = tensor.max().item<float>();
        tensor = (tensor - mn) / (mx - mn + 1e-6f);

        return {tensor, torch::tensor(lbl, torch::kInt64)};
    }

    torch::optional<size_t> size() const override {
        return samples_.size();
    }
};
