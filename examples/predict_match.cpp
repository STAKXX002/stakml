#include "stakml/dataset.hpp"
#include "stakml/tensor.hpp"
#include "stakml/nn.hpp"
#include "stakml/ops.hpp"
#include "stakml/serialize.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace stakml;

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Match Predictor AI (Elo-Powered)\n";
    std::cout << "══════════════════════════════════════════\n\n";

    std::cout << "Loading dataset and trained brain...\n";
    auto db = dataset::Football::load("../data/results.csv");
    size_t num_teams = db.id_to_team.size();
    size_t total_features = 2 * num_teams + 6;

    nn::Sequential model({
        std::make_shared<nn::Linear>(total_features, 256),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(256, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 3)
    });

    try {
        serialize::load_model(model, "world_cup.stak");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model. Did you run world_cup first?\n";
        return 1;
    }

    std::cout << "Ready.\n\n";

    // ── Interactive Loop ──
    while (true) {
        std::string team_a, team_b;
        std::cout << "Enter Home Team (or 'exit'): ";
        std::getline(std::cin, team_a);
        if (team_a == "exit") break;

        std::cout << "Enter Away Team: ";
        std::getline(std::cin, team_b);

        if (db.team_to_id.find(team_a) == db.team_to_id.end() || 
            db.team_to_id.find(team_b) == db.team_to_id.end()) {
            std::cout << "[!] One or both teams not found in history. Try again.\n\n";
            continue;
        }

        int id_a = db.team_to_id[team_a];
        int id_b = db.team_to_id[team_b];

        // ── NEW ELO LOOKUP ──
        // Instantly grab their final Power Rating and Goal Averages!
        float elo_a = db.current_stats.at(id_a).elo;
        float elo_b = db.current_stats.at(id_b).elo;
        float goals_a = db.current_stats.at(id_a).ema_goals;
        float goals_b = db.current_stats.at(id_b).ema_goals;

        // Build the Input Tensor
        auto X = std::make_shared<Tensor>(std::vector<size_t>{1, total_features});
        float* row_ptr = X->raw_ptr();
        std::fill(row_ptr, row_ptr + total_features, 0.0f);
        
        row_ptr[id_a] = 1.0f;
        row_ptr[num_teams + id_b] = 1.0f;
        
        size_t offset = 2 * num_teams;
        row_ptr[offset + 0] = 1.0f; // Neutral site
        row_ptr[offset + 1] = 1.0f; // World Cup Tournament Weight
        row_ptr[offset + 2] = elo_a / 2000.0f; // Normalized Elo
        row_ptr[offset + 3] = elo_b / 2000.0f;
        row_ptr[offset + 4] = goals_a / 5.0f; 
        row_ptr[offset + 5] = goals_b / 5.0f;

        // Forward Pass
        auto logits = model.forward(X);
        
        // Manual Softmax
        float max_logit = std::max({logits.at({0, 0}), logits.at({0, 1}), logits.at({0, 2})});
        float sum_exp = 0.0f;
        float probs[3];
        for (int i = 0; i < 3; ++i) {
            probs[i] = std::exp(logits.at({0, (size_t)i}) - max_logit);
            sum_exp += probs[i];
        }
        for (int i = 0; i < 3; ++i) probs[i] = (probs[i] / sum_exp) * 100.0f;

        // Print Results
        std::cout << "\n──────────────────────────────────────────\n";
        std::cout << " PREDICTION: " << team_a << " vs " << team_b << "\n";
        std::cout << " (" << team_a << " Elo: " << (int)elo_a << " | " << team_b << " Elo: " << (int)elo_b << ")\n";
        std::cout << "──────────────────────────────────────────\n";
        std::cout << team_a << " Win: " << std::fixed << std::setprecision(1) << probs[0] << "%\n";
        std::cout << "Draw:        " << probs[2] << "%\n";
        std::cout << team_b << " Win: " << probs[1] << "%\n\n";
    }

    return 0;
}