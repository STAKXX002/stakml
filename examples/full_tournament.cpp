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
#include <random>

using namespace stakml;

struct Standing {
    std::string team;
    int points = 0;
};

// ── CORE MATCH LOGIC ──────────────────────────────────────────────────────
std::pair<int, int> play_group_match(nn::Sequential& model, const dataset::Football& db, const std::string& team_a, const std::string& team_b, size_t total_features, size_t num_teams) {
    int id_a = db.team_to_id.at(team_a); int id_b = db.team_to_id.at(team_b);
    float elo_a = db.current_stats.at(id_a).elo; float elo_b = db.current_stats.at(id_b).elo;
    float goals_a = db.current_stats.at(id_a).ema_goals; float goals_b = db.current_stats.at(id_b).ema_goals;

    auto X = std::make_shared<Tensor>(std::vector<size_t>{1, total_features});
    float* ptr = X->raw_ptr();
    std::fill(ptr, ptr + total_features, 0.0f);
    ptr[id_a] = 1.0f; ptr[num_teams + id_b] = 1.0f;
    ptr[2 * num_teams] = 1.0f; ptr[2 * num_teams + 1] = 1.0f; 
    ptr[2 * num_teams + 2] = elo_a / 2000.0f; ptr[2 * num_teams + 3] = elo_b / 2000.0f;
    ptr[2 * num_teams + 4] = goals_a / 5.0f;  ptr[2 * num_teams + 5] = goals_b / 5.0f;

    auto logits = model.forward(X);
    float max_val = logits.at({0, 0}); int outcome = 0;
    if (logits.at({0, 1}) > max_val) { max_val = logits.at({0, 1}); outcome = 1; }
    if (logits.at({0, 2}) > max_val) { outcome = 2; }

    if (outcome == 0) return {3, 0}; else if (outcome == 1) return {0, 3}; else return {1, 1};
}

std::string simulate_knockout_match(nn::Sequential& model, const dataset::Football& db, const std::string& team_a, const std::string& team_b, size_t total_features, size_t num_teams) {
    int id_a = db.team_to_id.at(team_a); int id_b = db.team_to_id.at(team_b);
    float elo_a = db.current_stats.at(id_a).elo; float elo_b = db.current_stats.at(id_b).elo;
    float goals_a = db.current_stats.at(id_a).ema_goals; float goals_b = db.current_stats.at(id_b).ema_goals;

    auto X = std::make_shared<Tensor>(std::vector<size_t>{1, total_features});
    float* ptr = X->raw_ptr(); std::fill(ptr, ptr + total_features, 0.0f);
    ptr[id_a] = 1.0f; ptr[num_teams + id_b] = 1.0f;
    ptr[2 * num_teams] = 1.0f; ptr[2 * num_teams + 1] = 1.0f; 
    ptr[2 * num_teams + 2] = elo_a / 2000.0f; ptr[2 * num_teams + 3] = elo_b / 2000.0f;
    ptr[2 * num_teams + 4] = goals_a / 5.0f; ptr[2 * num_teams + 5] = goals_b / 5.0f;

    auto logits = model.forward(X);
    return (logits.at({0, 0}) > logits.at({0, 1})) ? team_a : team_b;
}

std::vector<Standing> simulate_group(nn::Sequential& model, const dataset::Football& db, std::vector<std::string> teams, size_t total_features, size_t num_teams) {
    std::vector<Standing> standings = {{teams[0], 0}, {teams[1], 0}, {teams[2], 0}, {teams[3], 0}};
    for (size_t i = 0; i < teams.size(); ++i) {
        for (size_t j = i + 1; j < teams.size(); ++j) {
            auto pts = play_group_match(model, db, teams[i], teams[j], total_features, num_teams);
            standings[i].points += pts.first; standings[j].points += pts.second;
        }
    }
    std::sort(standings.begin(), standings.end(), [](const Standing& a, const Standing& b) { return a.points > b.points; });
    return standings;
}


int main() {
    auto db = dataset::Football::load("../data/results.csv");
    size_t num_teams = db.id_to_team.size();
    size_t total_features = 2 * num_teams + 6;

    std::map<std::string, std::string> t_map = {
        {"USA", "United States"}, {"Czechia", "Czech Republic"}, {"Korea Republic", "South Korea"},
        {"Türkiye", "Turkey"}, {"Côte d'Ivoire", "Ivory Coast"}, {"IR Iran", "Iran"},
        {"Cabo Verde", "Cape Verde"}, {"Congo DR", "DR Congo"}
    };
    auto resolve = [&](const std::string& name) { return t_map.count(name) ? t_map[name] : name; };

    nn::Sequential model({
        std::make_shared<nn::Linear>(total_features, 256), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(256, 128), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 3)
    });
    serialize::load_model(model, "world_cup.stak");

    // ── 1. SILENT SIMULATION PHASE ──
    std::vector<std::vector<std::string>> group_fixtures = {
        {"Mexico", "South Africa", resolve("Korea Republic"), resolve("Czechia")},
        {"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"},
        {"Brazil", "Morocco", "Haiti", "Scotland"},
        {resolve("USA"), "Paraguay", "Australia", resolve("Türkiye")},
        {resolve("Côte d'Ivoire"), "Ecuador", "Germany", "Curaçao"},
        {"Netherlands", "Japan", "Sweden", "Tunisia"},
        {resolve("IR Iran"), "New Zealand", "Belgium", "Egypt"},
        {"Saudi Arabia", "Uruguay", "Spain", resolve("Cabo Verde")},
        {"France", "Senegal", "Iraq", "Norway"},
        {"Argentina", "Algeria", "Austria", "Jordan"},
        {"Portugal", resolve("Congo DR"), "Uzbekistan", "Colombia"},
        {"Ghana", "Panama", "England", "Croatia"}
    };
    
    std::vector<std::string> bracket;
    std::vector<Standing> all_third_places;
    std::vector<std::pair<std::string, std::string>> group_winners;

    for (size_t i = 0; i < 12; ++i) {
        auto standings = simulate_group(model, db, group_fixtures[i], total_features, num_teams);
        bracket.push_back(standings[0].team); bracket.push_back(standings[1].team); 
        group_winners.push_back({standings[0].team, standings[1].team});
        all_third_places.push_back(standings[2]); 
    }

    std::sort(all_third_places.begin(), all_third_places.end(), [](const Standing& a, const Standing& b) { return a.points > b.points; });
    std::vector<std::string> wildcards;
    for (int i = 0; i < 8; ++i) {
        bracket.push_back(all_third_places[i].team);
        wildcards.push_back(all_third_places[i].team);
    }

    std::mt19937 g(42); 
    std::shuffle(bracket.begin(), bracket.end(), g);

    std::vector<std::vector<std::string>> tournament_history;
    tournament_history.push_back(bracket); 
    std::vector<std::string> round_names = {"ROUND OF 32", "ROUND OF 16", "QUARTER-FINALS", "SEMI-FINALS", "THE FINAL"};

    for (size_t r = 0; r < round_names.size(); ++r) {
        std::vector<std::string> next_round;
        for (size_t i = 0; i < bracket.size(); i += 2) {
            std::string winner = simulate_knockout_match(model, db, bracket[i], bracket[i+1], total_features, num_teams);
            next_round.push_back(winner);
        }
        tournament_history.push_back(next_round); 
        bracket = next_round;
        if (bracket.size() == 1) break; 
    }

    std::vector<std::string> sf_teams = tournament_history[3]; 
    std::vector<std::string> finalists = tournament_history[4]; 
    std::vector<std::string> bronze_teams;
    for (const auto& team : sf_teams) {
        if (std::find(finalists.begin(), finalists.end(), team) == finalists.end()) bronze_teams.push_back(team); 
    }
    std::string bronze_winner = simulate_knockout_match(model, db, bronze_teams[0], bronze_teams[1], total_features, num_teams);

    // ── 2. COMPACT CLI TABLE RENDERER (NO EMOJIS) ──
    auto pad = [](const std::string& s, size_t w) {
        size_t len = s.length();
        return s + std::string(w > len ? w - len : 0, ' ');
    };

    std::cout << "\n\033[1;36m================================================================================\033[0m\n";
    std::cout << "\033[1;37m                    STAKML 2026 WORLD CUP: COMPACT DASHBOARD                    \033[0m\n";
    std::cout << "\033[1;36m================================================================================\033[0m\n\n";

    // GROUP STAGE COMPACT GRID
    std::cout << "\033[1;33m[ PHASE 1: GROUP STAGE ADVANCERS ]\033[0m\n";
    for (size_t i = 0; i < 12; i += 2) {
        std::cout << "  \033[1;30mGrp " << (char)('A' + i) << " │\033[0m  " 
                  << pad(group_winners[i].first, 18) 
                  << "\033[1;30m&\033[0m  " << pad(group_winners[i].second, 18)
                  << "\033[1;30m║ Grp " << (char)('A' + i + 1) << " │\033[0m  " 
                  << pad(group_winners[i+1].first, 18) 
                  << "\033[1;30m&\033[0m  " << group_winners[i+1].second << "\n";
    }

    // WILDCARDS COMPACT GRID
    std::cout << "\n\033[1;33m[ WILDCARD QUALIFIERS ]\033[0m\n  ";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << pad(wildcards[i], 20);
        if (i == 3) std::cout << "\n  ";
    }
    std::cout << "\n\n";

    // KNOCKOUT TABLES
    for (size_t r = 0; r < round_names.size(); ++r) {
        std::cout << "\033[1;36m[ " << round_names[r] << " ]\033[0m\n";
        for (size_t i = 0; i < tournament_history[r].size(); i += 2) {
            std::string t1 = tournament_history[r][i];
            std::string t2 = tournament_history[r][i+1];
            std::string w = tournament_history[r+1][i/2];
            
            std::cout << "  " << pad(t1, 24) 
                      << " \033[1;30mvs\033[0m  " 
                      << pad(t2, 24) 
                      << " \033[1;30m│ ➔\033[0m  " 
                      << "\033[1;32m" << w << "\033[0m\n";
        }
        std::cout << "\n";
    }

    // BRONZE MATCH
    std::cout << "\033[1;33m[ THIRD-PLACE PLAY-OFF ]\033[0m\n";
    std::cout << "  " << pad(bronze_teams[0], 24) 
              << " \033[1;30mvs\033[0m  " 
              << pad(bronze_teams[1], 24) 
              << " \033[1;30m│ ➔\033[0m  " 
              << "\033[1;33m" << bronze_winner << " (Bronze)\033[0m\n\n";

    // CHAMPION
    std::cout << "\033[1;36m================================================================================\033[0m\n";
    
    // Auto-center the winning team's name
    std::string champ_str = "WORLD CHAMPION: " + tournament_history[5][0];
    int pad1 = (80 - champ_str.length()) / 2;
    
    std::cout << std::string(pad1, ' ') << "\033[1;33m" << champ_str << "\033[0m\n";
    std::cout << "\033[1;36m================================================================================\033[0m\n\n";

    return 0;
}