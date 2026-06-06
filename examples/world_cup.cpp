#include "stakml/dataset.hpp"
#include "stakml/tensor.hpp"
#include "stakml/nn.hpp"
#include "stakml/ops.hpp"
#include "stakml/loss.hpp"
#include "stakml/optim.hpp"
#include "stakml/serialize.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace stakml;

// ── Feature Extractor ────────────────────────────────────────────────────────
// Converts a single FootballMatch into a flat array of floats for the Neural Net
void fill_feature_row(float* row_ptr, const dataset::FootballMatch& m, size_t num_teams) {
    size_t total_features = 2 * num_teams + 6;
    std::fill(row_ptr, row_ptr + total_features, 0.0f); 
    
    row_ptr[m.home_team_id] = 1.0f;
    row_ptr[num_teams + m.away_team_id] = 1.0f;
    
    size_t offset = 2 * num_teams;
    row_ptr[offset + 0] = m.is_neutral;
    row_ptr[offset + 1] = m.tournament_weight;
    
    // Normalize Elo (max is roughly ~2200, so dividing by 2000 keeps it near 1.0)
    row_ptr[offset + 2] = m.home_elo / 2000.0f;
    row_ptr[offset + 3] = m.away_elo / 2000.0f;
    
    // Normalize goals
    row_ptr[offset + 4] = m.home_ema_goals / 5.0f; 
    row_ptr[offset + 5] = m.away_ema_goals / 5.0f;
}

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Sports Betting Model\n";
    std::cout << "══════════════════════════════════════════\n\n";

    std::cout << "1. Loading & Engineering Data...\n";
    auto db = dataset::Football::load("../data/results.csv");
    size_t num_teams = db.id_to_team.size();
    size_t total_features = 2 * num_teams + 6;

    // Chronological Train/Test Split (80% Train, 20% Test)
    size_t train_size = static_cast<size_t>(db.matches.size() * 0.8);
    size_t test_size = db.matches.size() - train_size;
    std::cout << "   Training on matches 1872 - ~2000 (" << train_size << " games)\n";
    std::cout << "   Testing on matches ~2000 - 2024 (" << test_size << " games)\n\n";

    std::cout << "2. Building Architecture...\n";
    std::cout << "   Input Nodes: " << total_features << "\n";
    std::cout << "   Output Nodes: 3 (Home Win, Away Win, Draw)\n\n";

    nn::Sequential model({
        std::make_shared<nn::Linear>(total_features, 256),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(256, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 3)
    });

    // We use a smaller learning rate (0.01) because sports data is highly noisy
    optim::SGD opt(model.parameters(), 0.01f); 

    size_t batch_size = 64;
    size_t epochs = 10;
    size_t train_batches = train_size / batch_size;
    size_t test_batches  = test_size / batch_size;

    std::cout << "Starting Training...\n";
    std::cout << "───────────────────────────────────────────────────────────────────\n";

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        int train_correct = 0;

        // ── TRAINING ──
        for (size_t b = 0; b < train_batches; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{batch_size, total_features});
            std::vector<int> Y_batch(batch_size);
            float* x_ptr = X_batch->raw_ptr();

            for (size_t i = 0; i < batch_size; ++i) {
                const auto& match = db.matches[b * batch_size + i];
                fill_feature_row(x_ptr + (i * total_features), match, num_teams);
                Y_batch[i] = match.outcome;
            }

            opt.zero_grad();
            auto logits = model.forward(X_batch);
            auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));
            
            total_loss += ops::nll_loss(log_probs, Y_batch);
            train_correct += static_cast<int>(ops::accuracy(logits, Y_batch) * batch_size);

            log_probs.backward();
            opt.step();
        }

        // ── TESTING (Modern Era Matches) ──
        int test_correct = 0;
        for (size_t b = 0; b < test_batches; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{batch_size, total_features});
            std::vector<int> Y_batch(batch_size);
            float* x_ptr = X_batch->raw_ptr();

            for (size_t i = 0; i < batch_size; ++i) {
                const auto& match = db.matches[train_size + (b * batch_size + i)];
                fill_feature_row(x_ptr + (i * total_features), match, num_teams);
                Y_batch[i] = match.outcome;
            }

            auto logits = model.forward(X_batch);
            test_correct += static_cast<int>(ops::accuracy(logits, Y_batch) * batch_size);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        float avg_loss = total_loss / train_batches;
        float train_acc = static_cast<float>(train_correct) / (train_batches * batch_size) * 100.0f;
        float test_acc  = static_cast<float>(test_correct) / (test_batches * batch_size) * 100.0f;

        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss 
                  << " | Train Acc: " << std::fixed << std::setprecision(2) << train_acc << "%"
                  << " | Test Acc: " << std::fixed << std::setprecision(2) << test_acc << "%"
                  << " | Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s\n";
    }

    std::cout << "───────────────────────────────────────────────────────────────────\n";
    std::cout << "Saving model to 'world_cup.stak'...\n";
    serialize::save_model(model, "world_cup.stak");
    std::cout << "Run complete!\n";

    return 0;
}