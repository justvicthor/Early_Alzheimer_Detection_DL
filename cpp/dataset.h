// dataset.h
#pragma once
#include "config.h"
#include <torch/torch.h>
#include <nifti1_io.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <random>
#include <iostream>

namespace fs = std::filesystem;
namespace F  = torch::nn::functional;

/* 3D Gaussian blur --------------------------------------------------------- */
inline torch::Tensor gaussian_blur_3d(torch::Tensor img, double sigma)
{
    if (sigma < 1e-6) return img;

    int k = static_cast<int>(2 * std::ceil(2 * sigma) + 1);
    if (k % 2 == 0) ++k;
    auto rng   = torch::arange(k, img.options()) - (k - 1) / 2.0;
    auto ker1d = torch::exp(-0.5 * torch::pow(rng / sigma, 2));
    ker1d /= ker1d.sum();

    auto ker3d = torch::einsum("i,j,k->ijk",{ker1d,ker1d,ker1d})
                    .view({1,1,k,k,k});

    int pad = (k-1)/2;
    img = img.unsqueeze(0).unsqueeze(0);
    auto out = F::conv3d(img, ker3d, F::Conv3dFuncOptions().padding(pad));
    return out.squeeze();
}

/* ADNI Dataset Loader --------------------------------------------------------- */
class ADNIDataset
        : public torch::data::datasets::Dataset<ADNIDataset,torch::data::Example<>>
{
public:
    ADNIDataset(const Config& cfg, std::string mode)
        : cfg_(cfg), mode_(std::move(mode)),
          rng_(cfg.seed),
          sigma_dist_(cfg.blur_sigma.at(0), cfg.blur_sigma.at(1))
    {
        /* ------------  Create label → index mapping ------------ */
        std::vector<std::string> labs = (cfg.num_classes==3)?
            std::vector<std::string>{"CN","LMCI","AD"} :
            std::vector<std::string>{"CN","AD"};

        for (size_t i=0;i<labs.size();++i) label_map_[labs[i]] = i;

        /* ------------  Read subject list from TSV ------------ */
        const std::string& tsv = (mode_=="train")? cfg.train_tsv
                                 :(mode_=="val")? cfg.val_tsv : cfg.test_tsv;

        std::ifstream fin(tsv);
        if (!fin) throw std::runtime_error("Cannot open "+tsv);
        std::string line; std::getline(fin,line);          // skip header

        while (std::getline(fin,line)) {
            std::stringstream ss(line);
            std::vector<std::string> col;
            for (std::string field; std::getline(ss,field,'\t');) col.push_back(field);
            if (col.size()<11) continue;

            const std::string& sid  = col[0];
            const std::string& pid  = col[1];
            const std::string& diag = col[10];

            auto it = label_map_.find(diag);
            if (it==label_map_.end()) continue;

            fs::path dir = fs::path(cfg.scans_dir)/"subjects"/pid/sid/
                           "t1/spm/segmentation/normalized_space";
            if (!fs::exists(dir)) continue;

            for (auto& e : fs::directory_iterator(dir)) {
                if (e.path().filename().string().find("Space_T1w")!=std::string::npos){
                    samples_.emplace_back(e.path().string(), it->second);
                    break;
                }
            }
        }
        std::cout<<"["<<mode_<<"] samples: "<<samples_.size()<<"\n";
    }

    /* Load and return a sample (volume + label) --------------------------------------------------------- */
    torch::data::Example<> get(size_t idx) override
    {
        auto [path,label] = samples_.at(idx);

        // load and decode NIfTI image
        nifti_image* img = nifti_image_read(path.c_str(),1);
        if(!img) throw std::runtime_error("NIfTI read failed: "+path);

        size_t nvox = img->nvox;
        std::vector<float> buf(nvox);
        float s = (img->scl_slope!=0)? img->scl_slope:1.0f;
        float i = img->scl_inter;

#define COPY_AS(type) {auto p=reinterpret_cast<type*>(img->data);\
                       for(size_t k=0;k<nvox;++k) buf[k]=s*p[k]+i;}

        switch(img->datatype){
            case DT_UINT8:   COPY_AS(uint8_t);  break;
            case DT_INT16:   COPY_AS(int16_t);  break;
            case DT_INT32:   COPY_AS(int32_t);  break;
            case DT_FLOAT32: COPY_AS(float);    break;
            case DT_FLOAT64: COPY_AS(double);   break;
            default: nifti_image_free(img);
                     throw std::runtime_error("Unsupported NIfTI type");
        }
#undef COPY_AS
        int X=img->nx, Y=img->ny, Z=img->nz;
        nifti_image_free(img);

        auto t = torch::from_blob(buf.data(),{Z,Y,X},torch::kFloat32).clone();
        t = (t - t.min()) / (t.max()-t.min()+1e-6);
        t = torch::nan_to_num(t);

        // Apply augmentation or center crop
        if (mode_=="train" && cfg_.use_augmentation)
        {
            if (std::uniform_real_distribution<>(0,1)(rng_)<0.5)
                t = gaussian_blur_3d(t, sigma_dist_(rng_));
            t = random_crop(t, cfg_.crop_size);
        }
        else
            t = center_crop(t, cfg_.crop_size);

        return { t.unsqueeze(0), torch::tensor(label, torch::kInt64) };
    }

    // Return total number of samples
    torch::optional<size_t> size() const override { return samples_.size(); }

private:
    // Random Cropping helper
    static torch::Tensor random_crop(torch::Tensor v,int sz)
    {
        auto [d,h,w] = std::tuple<int,int,int>{v.size(0),v.size(1),v.size(2)};
        int d0 = std::rand()%(d-sz+1);
        int h0 = std::rand()%(h-sz+1);
        int w0 = std::rand()%(w-sz+1);
        return v.slice(0,d0,d0+sz)
                .slice(1,h0,h0+sz)
                .slice(2,w0,w0+sz);
    }
    // Center Cropping helper
    static torch::Tensor center_crop(torch::Tensor v,int sz)
    {
        auto [d,h,w] = std::tuple<int,int,int>{v.size(0),v.size(1),v.size(2)};
        int d0=(d-sz)/2, h0=(h-sz)/2, w0=(w-sz)/2;
        return v.slice(0,d0,d0+sz)
                .slice(1,h0,h0+sz)
                .slice(2,w0,w0+sz);
    }

    // Dataset fields
    const Config& cfg_;
    std::string   mode_;
    std::vector<std::pair<std::string,int64_t>> samples_;
    std::unordered_map<std::string,int64_t>     label_map_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> sigma_dist_;
};
