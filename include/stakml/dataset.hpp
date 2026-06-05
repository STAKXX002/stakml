#pragma once
#include "tensor.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// dataset.hpp — Data loading utilities
//
// Efficiency Note:
//   MNIST files are binary. To save time and CPU cycles, we read the entire
//   file block into a std::vector<uint8_t> in a single I/O operation, rather 
//   than looping over fstream::read(). We then convert bytes to floats.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace dataset {

// Helper to swap endianness (MNIST files are Big-Endian, most modern CPUs are Little-Endian)
inline uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

struct MNIST {
    Tensor images;           // Shape: {num_images, 784}
    std::vector<int> labels; // Shape: {num_images}
    size_t num_samples;

    static MNIST load(const std::string& image_path, const std::string& label_path) {
        MNIST dataset;

        // ─── 1. Load Labels ──────────────────────────────────────────────────
        std::ifstream label_file(label_path, std::ios::binary);
        if (!label_file) throw std::runtime_error("Cannot open " + label_path);

        uint32_t magic, num_labels;
        label_file.read(reinterpret_cast<char*>(&magic), 4);
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
        magic = swap_endian(magic);
        num_labels = swap_endian(num_labels);

        if (magic != 2049) throw std::runtime_error("Invalid MNIST label file magic number");

        dataset.num_samples = num_labels;
        dataset.labels.resize(num_labels);
        
        // Single block read for all labels
        std::vector<uint8_t> raw_labels(num_labels);
        label_file.read(reinterpret_cast<char*>(raw_labels.data()), num_labels);
        for (size_t i = 0; i < num_labels; ++i) {
            dataset.labels[i] = static_cast<int>(raw_labels[i]);
        }

        // ─── 2. Load Images ──────────────────────────────────────────────────
        std::ifstream image_file(image_path, std::ios::binary);
        if (!image_file) throw std::runtime_error("Cannot open " + image_path);

        uint32_t num_images, rows, cols;
        image_file.read(reinterpret_cast<char*>(&magic), 4);
        image_file.read(reinterpret_cast<char*>(&num_images), 4);
        image_file.read(reinterpret_cast<char*>(&rows), 4);
        image_file.read(reinterpret_cast<char*>(&cols), 4);

        magic = swap_endian(magic);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        if (magic != 2051) throw std::runtime_error("Invalid MNIST image file magic number");
        if (num_images != num_labels) throw std::runtime_error("Mismatch between image and label counts");

        size_t image_size = rows * cols; // Should be 28x28 = 784
        dataset.images = Tensor({num_images, image_size});

        // Single massive block read for all image data
        std::vector<uint8_t> raw_images(num_images * image_size);
        image_file.read(reinterpret_cast<char*>(raw_images.data()), num_images * image_size);

        // Convert to float in [0.0, 1.0] range for neural network stability
        float* img_ptr = dataset.images.raw_ptr();
        for (size_t i = 0; i < num_images * image_size; ++i) {
            img_ptr[i] = static_cast<float>(raw_images[i]) / 255.0f;
        }

        return dataset;
    }
};

struct TeamStats
{
    float elo = 1500.0f;    // Everyone starts at 1500
    float ema_goals = 1.0f; // Baseline average of 1 goal

    void update_elo(float actual_score, float expected_score, float tournament_weight)
    {
        // K-factor dictates how much a match changes a rating.
        // Friendlies = 20, World Cup = 60
        float k_factor = 20.0f + (tournament_weight * 40.0f);
        elo += k_factor * (actual_score - expected_score);
    }

    void update_goals(int goals_scored)
    {
        // 20% newest match, 80% historical memory
        ema_goals = (0.2f * goals_scored) + (0.8f * ema_goals);
    }
};

struct FootballMatch
{
    int home_team_id;
    int away_team_id;
    int outcome;

    float is_neutral;
    float tournament_weight;

    // The New High-Fidelity Features
    float home_elo;
    float away_elo;
    float home_ema_goals;
    float away_ema_goals;
};

struct Football
{
    std::vector<FootballMatch> matches;
    std::unordered_map<std::string, int> team_to_id;
    std::vector<std::string> id_to_team;
    std::unordered_map<int, TeamStats> current_stats; // Store final stats for inference later!

    int get_or_add_team(const std::string &name)
    {
        auto it = team_to_id.find(name);
        if (it != team_to_id.end())
            return it->second;
        int new_id = id_to_team.size();
        team_to_id[name] = new_id;
        id_to_team.push_back(name);
        return new_id;
    }

    static Football load(const std::string &filepath)
    {
        Football dataset;
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Could not open " + filepath);

        std::string line;
        std::getline(file, line);

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string date, home, away, h_score_str, a_score_str, tournament, city, country, neutral_str;

            std::getline(ss, date, ',');
            std::getline(ss, home, ',');
            std::getline(ss, away, ',');
            std::getline(ss, h_score_str, ',');
            std::getline(ss, a_score_str, ',');
            std::getline(ss, tournament, ',');
            std::getline(ss, city, ',');
            std::getline(ss, country, ',');
            std::getline(ss, neutral_str, '\r');

            if (h_score_str.empty() || a_score_str.empty())
                continue;

            int h_score = 0, a_score = 0;
            try
            {
                h_score = std::stoi(h_score_str);
                a_score = std::stoi(a_score_str);
            }
            catch (...)
            {
                continue;
            }

            int h_id = dataset.get_or_add_team(home);
            int a_id = dataset.get_or_add_team(away);

            FootballMatch m;
            m.home_team_id = h_id;
            m.away_team_id = a_id;

            if (h_score > a_score)
                m.outcome = 0;
            else if (a_score > h_score)
                m.outcome = 1;
            else
                m.outcome = 2;

            m.is_neutral = (neutral_str == "TRUE") ? 1.0f : 0.0f;
            if (tournament == "Friendly")
                m.tournament_weight = 0.0f;
            else if (tournament == "FIFA World Cup")
                m.tournament_weight = 1.0f;
            else
                m.tournament_weight = 0.5f;

            // 1. Snapshot the pre-match Elo and Form for the Neural Net
            m.home_elo = dataset.current_stats[h_id].elo;
            m.away_elo = dataset.current_stats[a_id].elo;
            m.home_ema_goals = dataset.current_stats[h_id].ema_goals;
            m.away_ema_goals = dataset.current_stats[a_id].ema_goals;

            dataset.matches.push_back(m);

            // 2. Post-Match Elo Math Update
            float expected_h = 1.0f / (1.0f + std::pow(10.0f, (m.away_elo - m.home_elo) / 400.0f));
            float expected_a = 1.0f - expected_h;

            float score_h = (m.outcome == 0) ? 1.0f : (m.outcome == 2 ? 0.5f : 0.0f);
            float score_a = (m.outcome == 1) ? 1.0f : (m.outcome == 2 ? 0.5f : 0.0f);

            dataset.current_stats[h_id].update_elo(score_h, expected_h, m.tournament_weight);
            dataset.current_stats[a_id].update_elo(score_a, expected_a, m.tournament_weight);
            dataset.current_stats[h_id].update_goals(h_score);
            dataset.current_stats[a_id].update_goals(a_score);
        }
        return dataset;
    }
};

} // namespace dataset
} // namespace stakml