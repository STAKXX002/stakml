#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/loss.hpp"
#include "stakml/optim.hpp"
#include <iostream>
#include <string>
#include <cmath>

using namespace stakml;

static int pass_count = 0;
static int fail_count = 0;

#define RUN_TEST(fn) do {                           \
    try {                                           \
        fn();                                       \
        std::cout << "  [PASS] " #fn "\n";          \
        ++pass_count;                               \
    } catch (std::exception& e) {                   \
        std::cout << "  [FAIL] " #fn "\n"           \
                  << "        " << e.what() <<"\n"; \
        ++fail_count;                               \
    }                                               \
} while(0)

#define ASSERT(cond) do {                           \
    if (!(cond)) throw std::runtime_error(          \
        "Assertion failed: " #cond                  \
        " at line " + std::to_string(__LINE__));    \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

void test_sgd_updates_weights() {
    nn::Linear fc(2, 2);
    optim::SGD opt(fc.parameters(), 0.1f);

    // Save initial state of the first weight
    float orig_w = fc.W->raw_ptr()[0];

    // Forward pass with ones
    auto x = std::make_shared<Tensor>(Tensor::ones({1, 2}));
    Tensor out = fc.forward(x);

    // Backward pass: call it directly on 'out'
    out.backward();

    // Verify gradients populated
    ASSERT(fc.W->grad_ != nullptr);
    ASSERT(fc.W->grad_->raw_ptr()[0] != 0.0f);

    // Take an optimization step
    opt.step();

    // Verify weights changed
    float new_w = fc.W->raw_ptr()[0];
    ASSERT(orig_w != new_w);

    // Verify zero_grad clears the gradients
    opt.zero_grad();
    ASSERT(fc.W->grad_->raw_ptr()[0] == 0.0f);
}

void test_overfit_single_batch() {
    // 1. Setup a mini-MLP using the new Sequential API
    nn::Sequential model({
        std::make_shared<nn::Linear>(4, 8),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(8, 3)
    });

    optim::SGD opt(model.parameters(), 0.1f); // 0.1 LR

    // 2. Dummy dataset (Batch of 4 items, 4 features)
    auto X = std::make_shared<Tensor>(Tensor::randn({4, 4}));
    std::vector<int> Y = {0, 1, 2, 0}; // Target classes

    // 3. --- STEP 1 ---
    auto logits1 = model.forward(X);
    auto log_probs1 = ops::log_softmax(std::make_shared<Tensor>(logits1));
    float loss1 = ops::nll_loss(log_probs1, Y);

    log_probs1.backward();
    opt.step();

    // 4. --- STEP 2 ---
    opt.zero_grad(); // Crucial: clear old gradients!
    
    auto logits2 = model.forward(X);
    auto log_probs2 = ops::log_softmax(std::make_shared<Tensor>(logits2));
    float loss2 = ops::nll_loss(log_probs2, Y);

    // 5. Assert the network is actually learning
    std::cout << "    Loss step 1: " << loss1 << "\n";
    std::cout << "    Loss step 2: " << loss2 << "\n";
    ASSERT(loss2 < loss1);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Optimizer Tests\n";
    std::cout << "══════════════════════════════════════════\n\n";

    RUN_TEST(test_sgd_updates_weights);
    RUN_TEST(test_overfit_single_batch);

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  " << pass_count << " / " << (pass_count + fail_count) << " passed\n";
    std::cout << "══════════════════════════════════════════\n";

    return fail_count > 0 ? 1 : 0;
}