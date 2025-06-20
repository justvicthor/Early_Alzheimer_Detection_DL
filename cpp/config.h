// config.h
#pragma once
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

struct Config
{
    /* ---------- data ---------- */
    std::string scans_dir, train_tsv, val_tsv, test_tsv;
    int   batch_sz{}, val_batch_sz{}, test_batch_sz{}, workers{}, crop_size{};
    std::vector<double> blur_sigma;          
    bool  use_augmentation{true};

    /* ---------- model --------- */
    int in_channels{}, num_classes{}, expansion{}, feature_dim{}, nhid{};
    std::string norm_type{"Instance"};       // "Instance" | "Batch"

    /* ---------- training ------ */
    int    epochs{};
    double lr{}, grad_clip{-1.0};            
    int    seed{14};

    /* ---------- files --------- */
    std::string file_name, log_csv;
};

/*-------------------------------------------------------------------------*/
inline Config load_config(const std::string& yaml_path)
{
    YAML::Node y = YAML::LoadFile(yaml_path);
    Config c;

    /* --- data --- */
    const auto& d = y["data"];
    c.scans_dir       = d["scans_dir"].as<std::string>();
    c.train_tsv       = d["train_tsv"].as<std::string>();
    c.val_tsv         = d["val_tsv"].as<std::string>();
    c.test_tsv        = d["test_tsv"].as<std::string>();
    c.batch_sz        = d["batch_size"].as<int>();
    c.val_batch_sz    = d["val_batch_size"].as<int>();
    c.test_batch_sz   = d["test_batch_size"].as<int>();
    c.workers         = d["workers"].as<int>();
    c.crop_size       = d["crop_size"].as<int>();
    c.blur_sigma      = d["blur_sigma"].as<std::vector<double>>();
    c.use_augmentation= d["use_augmentation"].as<bool>();

    /* --- model --- */
    const auto& m   = y["model"];
    c.in_channels   = m["in_channels"].as<int>();
    c.num_classes   = m["num_classes"].as<int>();
    c.expansion     = m["expansion"].as<int>();
    c.feature_dim   = m["feature_dim"].as<int>();
    c.nhid          = m["nhid"].as<int>();
    c.norm_type     = m["norm_type"].as<std::string>();

    /* --- training --- */
    const auto& t   = y["training"];
    c.epochs        = t["epochs"].as<int>();
    c.lr            = t["optimizer"]["lr"].as<double>();
    if (!t["gradient_clip"].IsNull())
        c.grad_clip = t["gradient_clip"].as<double>();
    if (y["seed"])  c.seed = y["seed"].as<int>();

    /* --- files --- */
    c.file_name     = y["file_name"].as<std::string>();
    c.log_csv       = y["log_csv"].as<std::string>();

    return c;
}
