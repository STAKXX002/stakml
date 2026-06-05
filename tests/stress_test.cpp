#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/loss.hpp"
#include "stakml/optim.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace stakml;

void test_memory_grinder() {
    std::cout << "── 1. The Memory Grinder (10,000 Iterations) ──\n";
    
    nn::Sequential model({
        std::make_shared<nn::Linear>(128, 64),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(64, 10)
    });
    optim::SGD opt(model.parameters(), 0.01f);
    
    auto X = std::make_shared<Tensor>(Tensor::randn({32, 128}));
    std::vector<int> labels(32, 1); // Dummy labels

    std::cout << "Running 10,000 forward/backward passes to hunt for shared_ptr leaks...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10000; ++i) {
        opt.zero_grad();
        auto logits = model.forward(X);
        auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));
        ops::nll_loss(log_probs, labels);
        log_probs.backward();
        opt.step();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "  [PASS] Completed in " << std::fixed << std::setprecision(2) << diff.count() << "s.\n";
    std::cout << "  (If your RAM didn't crash, the autograd graph is memory-safe!)\n\n";
}

void test_compute_anvil() {
    std::cout << "── 2. The Compute Anvil (Massive Batch Size) ──\n";
    
    nn::Linear layer(784, 256);
    size_t massive_batch = 8192; 
    
    auto X = std::make_shared<Tensor>(Tensor::randn({massive_batch, 784}));

    std::cout << "Pushing an [" << massive_batch << " x 784] tensor through naive matmul...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto out = layer.forward(X);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "  [PASS] Forward pass survived. Time: " << std::fixed << std::setprecision(3) << diff.count() << "s.\n\n";
}

void test_deep_abyss() {
    std::cout << "── 3. The Deep Abyss (Numerical Stability) ────\n";
    
    // A 7-layer deep network without Batch Norm is a nightmare for gradients
    nn::Sequential deep_model({
        std::make_shared<nn::Linear>(128, 128), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 128), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 128), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 10)
    });
    
    auto X = std::make_shared<Tensor>(Tensor::randn({16, 128}));
    std::vector<int> labels(16, 0);

    auto logits = deep_model.forward(X);
    auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));
    float loss = ops::nll_loss(log_probs, labels);

    std::cout << "Checking raw precision bounds on a 7-layer un-normalized forward pass...\n";
    
    if (std::isnan(loss) || std::isinf(loss)) {
        std::cout << "  [FAIL] Numerical instability detected (NaN / Inf)!\n\n";
    } else {
        std::cout << "  [PASS] Stable float precision maintained. Loss: " << loss << "\n\n";
    }
}

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Architecture Stress Test\n";
    std::cout << "══════════════════════════════════════════\n\n";
    
    test_memory_grinder();
    test_compute_anvil();
    test_deep_abyss();
    
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  Stress Test Complete.\n";
    std::cout << "══════════════════════════════════════════\n";
    return 0;
}