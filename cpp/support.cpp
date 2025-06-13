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
                    // std::cout << "  ✓ trovato: " << e.path() << '\n'; // inghippo?
                    break;
                }
            }