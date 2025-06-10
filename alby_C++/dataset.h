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
    std::vector<std::pair<std::string, int64_t>> samples_;

public:
    explicit ADNIDataset(const std::string& scans_dir,
                         const std::string& tsv_path,
                         int num_classes = 3)
    {
        // Define valid labels
        std::vector<std::string> valid = (num_classes == 3)
            ? std::vector<std::string>{"CN", "LMCI", "AD"}
            : std::vector<std::string>{"CN", "AD"};

        std::unordered_map<std::string, int64_t> label_map;
        for (size_t i = 0; i < valid.size(); ++i)
            label_map[valid[i]] = static_cast<int64_t>(i);

        // Read TSV
        std::ifstream file(tsv_path);
        if (!file) throw std::runtime_error("Cannot open TSV: " + tsv_path);
        std::string line;
        std::getline(file, line);  // Skip header

        namespace fs = std::filesystem;
        while (std::getline(file, line)) {
            // Split the line by TAB
            std::vector<std::string> col;
            std::stringstream ls(line);
            std::string field;
            while (std::getline(ls, field, '\t'))
                col.push_back(field);
            if (col.size() < 11) continue;  // Malformed line

            // TSV columns
            std::string sid = col[0];   // session_id (e.g., ses-M012)
            std::string pid = col[1];   // participant_id (e.g., sub-ADNI...)
            std::string diag = col[10]; // diagnosis

            auto it = label_map.find(diag);
            if (it == label_map.end()) continue;  // Diagnosis not needed

            // Path to images
            fs::path p = fs::path(scans_dir) / "subjects" / pid / sid
                       / "t1" / "spm" / "segmentation" / "normalized_space";
            if (!fs::exists(p)) {
                std::cout << "‼  Dir mancante: " << p << '\n';
                continue;
            }

            for (auto& e : fs::directory_iterator(p)) {
                const std::string fn = e.path().filename().string();

                /* Want ONLY the structural volume:
                   - must contain "_T1w"
                   - must NOT contain "_segm", "probability", "transformation"
                   - extension must be .nii or .nii.gz */
                bool estensione = fn.ends_with(".nii") || fn.ends_with(".nii.gz");
                bool t1w        = fn.find("_T1w") != std::string::npos;
                bool not_aux    = (fn.find("_segm")          == std::string::npos) &&
                                  (fn.find("probability")    == std::string::npos) &&
                                  (fn.find("transformation") == std::string::npos);

                if (estensione && t1w && not_aux) {
                    samples_.emplace_back(e.path().string(), it->second);
                    std::cout << "  ✓ aggiunto: " << e.path() << '\n';
                    break;  // Found the right volume -> exit the for
                }
            }
        }

        // Summary
        if (samples_.empty())
            std::cout << "‼  Nessuna immagine trovata! (path/pattern errati?)\n";
        std::cout << "[ADNI] samples found: " << samples_.size() << '\n';
    }

    torch::data::Example<> get(size_t idx) override {
        auto [path, lbl] = samples_.at(idx);

        // Load NIfTI
        nifti_image* img = nifti_image_read(path.c_str(), 1);
        if (!img) throw std::runtime_error("Failed NIfTI read: " + path);

        // Convert to float applying scaling (scl_slope, scl_inter)
        size_t nvox = img->nvox;  // Total number of voxels
        std::vector<float> buffer(nvox);

        if (img->datatype == DT_UINT8) {
            auto raw = static_cast<uint8_t*>(img->data);
            for (size_t i = 0; i < nvox; ++i)
                buffer[i] = img->scl_slope * raw[i] + img->scl_inter;
        } else if (img->datatype == DT_INT16) {
            auto raw = static_cast<int16_t*>(img->data);
            for (size_t i = 0; i < nvox; ++i)
                buffer[i] = img->scl_slope * raw[i] + img->scl_inter;
        } else if (img->datatype == DT_INT32) {
            auto raw = static_cast<int32_t*>(img->data);
            for (size_t i = 0; i < nvox; ++i)
                buffer[i] = img->scl_slope * raw[i] + img->scl_inter;
        } else if (img->datatype == DT_FLOAT32) {
            auto raw = static_cast<float*>(img->data);
            for (size_t i = 0; i < nvox; ++i)
                buffer[i] = img->scl_slope * raw[i] + img->scl_inter;
        } else {
            throw std::runtime_error("Unsupported NIfTI datatype: " + std::to_string(img->datatype));
        }

        int X = img->nx, Y = img->ny, Z = img->nz;
        auto tensor = torch::from_blob(buffer.data(), {Z, Y, X}, torch::kFloat32).clone();
        nifti_image_free(img);  // Deallocate img after clone

        // Add channel dimension
        tensor = tensor.unsqueeze(0);  // (1, Z, Y, X)

        // Robust normalization
        auto stats = tensor.flatten();
        float mn = stats.min().item<float>();
        float mx = stats.max().item<float>();

        float range = mx - mn;
        if (std::isnan(mn) || std::isnan(mx) || range < 1e-8f) {
            tensor = torch::zeros_like(tensor);  // Problematic volume
        } else {
            tensor = (tensor - mn) / range;
        }

        tensor = torch::nan_to_num(tensor);  // Replace NaNs with 0

        return {tensor, torch::tensor(lbl, torch::kInt64)};
    }

    torch::optional<size_t> size() const override {
        return samples_.size();
    }
};
