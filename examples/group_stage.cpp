#include "stakml/dataset.hpp"
#include "stakml/tensor.hpp"
#include "stakml/nn.hpp"
#include "stakml/serialize.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace stakml;

struct Standing {
    std::string team;
    int points = 0;
};

// ── NEW ELO LOOKUP ──
std::pair<int, int> play_group_match(nn::Sequential& model, const dataset::Football& db, const std::string& team_a, const std::string& team_b, size_t total_features, size_t num_teams) {
    int id_a = db.team_to_id.at(team_a);
    int id_b = db.team_to_id.at(team_b);

    // Instantly grab the final historical Elo and EMA!
    float elo_a = db.current_stats.at(id_a).elo;
    float elo_b = db.current_stats.at(id_b).elo;
    float goals_a = db.current_stats.at(id_a).ema_goals;
    float goals_b = db.current_stats.at(id_b).ema_goals;

    auto X = std::make_shared<Tensor>(std::vector<size_t>{1, total_features});
    float* ptr = X->raw_ptr();
    std::fill(ptr, ptr + total_features, 0.0f);
    
    ptr[id_a] = 1.0f;
    ptr[num_teams + id_b] = 1.0f;
    ptr[2 * num_teams] = 1.0f;     // Neutral site
    ptr[2 * num_teams + 1] = 1.0f; // World Cup weight
    ptr[2 * num_teams + 2] = elo_a / 2000.0f;
    ptr[2 * num_teams + 3] = elo_b / 2000.0f;
    ptr[2 * num_teams + 4] = goals_a / 5.0f;
    ptr[2 * num_teams + 5] = goals_b / 5.0f;

    auto logits = model.forward(X);
    
    float max_val = logits.at({0, 0});
    int outcome = 0;
    if (logits.at({0, 1}) > max_val) { max_val = logits.at({0, 1}); outcome = 1; }
    if (logits.at({0, 2}) > max_val) { outcome = 2; }

    std::cout << "    " << team_a << " vs " << team_b << " -> ";
    if (outcome == 0) { std::cout << team_a << " Wins (3 pts)\n"; return {3, 0}; }
    else if (outcome == 1) { std::cout << team_b << " Wins (3 pts)\n"; return {0, 3}; }
    else { std::cout << "DRAW (1 pt each)\n"; return {1, 1}; }
}

void simulate_group(nn::Sequential& model, const dataset::Football& db, std::string group_name, std::vector<std::string> teams, size_t total_features, size_t num_teams) {
    std::cout << "\n═══ GROUP " << group_name << " ════════════════════════════\n";
    std::vector<Standing> standings = {{teams[0], 0}, {teams[1], 0}, {teams[2], 0}, {teams[3], 0}};

    for (size_t i = 0; i < teams.size(); ++i) {
        for (size_t j = i + 1; j < teams.size(); ++j) {
            auto pts = play_group_match(model, db, teams[i], teams[j], total_features, num_teams);
            standings[i].points += pts.first;
            standings[j].points += pts.second;
        }
    }

    std::sort(standings.begin(), standings.end(), [](const Standing& a, const Standing& b) {
        return a.points > b.points;
    });

    std::cout << "\n  [FINAL STANDINGS]\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  " << i+1 << ". " << std::left << std::setw(15) << standings[i].team << " : " << standings[i].points << " pts";
        if (i < 2) std::cout << " (ADVANCES)";
        std::cout << "\n";
    }
}

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — 2026 Group Stage (Elo Edition)\n";
    std::cout << "══════════════════════════════════════════\n";

    auto db = dataset::Football::load("../data/results.csv");
    size_t num_teams = db.id_to_team.size();
    size_t total_features = 2 * num_teams + 6;

    std::map<std::string, std::string> fifa_to_historical = {
        {"USA", "United States"}, {"Czechia", "Czech Republic"},
        {"Korea Republic", "South Korea"}, {"Türkiye", "Turkey"},
        {"Côte d'Ivoire", "Ivory Coast"}, {"IR Iran", "Iran"},
        {"Cabo Verde", "Cape Verde"}, {"Congo DR", "DR Congo"}
    };

    auto resolve = [&](const std::string& name) {
        return fifa_to_historical.count(name) ? fifa_to_historical[name] : name;
    };

    nn::Sequential model({
        std::make_shared<nn::Linear>(total_features, 256), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(256, 128), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 3)
    });
    serialize::load_model(model, "world_cup.stak");

    simulate_group(model, db, "A", {"Mexico", "South Africa", resolve("Korea Republic"), resolve("Czechia")}, total_features, num_teams);
    simulate_group(model, db, "B", {"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"}, total_features, num_teams);
    simulate_group(model, db, "C", {"Brazil", "Morocco", "Haiti", "Scotland"}, total_features, num_teams);
    simulate_group(model, db, "D", {resolve("USA"), "Paraguay", "Australia", resolve("Türkiye")}, total_features, num_teams);
    simulate_group(model, db, "E", {resolve("Côte d'Ivoire"), "Ecuador", "Germany", "Curaçao"}, total_features, num_teams);
    simulate_group(model, db, "F", {"Netherlands", "Japan", "Sweden", "Tunisia"}, total_features, num_teams);
    simulate_group(model, db, "G", {resolve("IR Iran"), "New Zealand", "Belgium", "Egypt"}, total_features, num_teams);
    simulate_group(model, db, "H", {"Saudi Arabia", "Uruguay", "Spain", resolve("Cabo Verde")}, total_features, num_teams);
    simulate_group(model, db, "I", {"France", "Senegal", "Iraq", "Norway"}, total_features, num_teams);
    simulate_group(model, db, "J", {"Argentina", "Algeria", "Austria", "Jordan"}, total_features, num_teams);
    simulate_group(model, db, "K", {"Portugal", resolve("Congo DR"), "Uzbekistan", "Colombia"}, total_features, num_teams);
    simulate_group(model, db, "L", {"Ghana", "Panama", "England", "Croatia"}, total_features, num_teams);

    return 0;
}