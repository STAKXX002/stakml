#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/loss.hpp"
#include "stakml/optim.hpp"
#include "stakml/dataset.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace stakml;

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — MNIST Training & Testing Loop\n";
    std::cout << "══════════════════════════════════════════\n\n";

    // 1. Load Datasets
    std::string train_img = "../data/train-images-idx3-ubyte";
    std::string train_lbl = "../data/train-labels-idx1-ubyte";
    std::string test_img  = "../data/t10k-images-idx3-ubyte";
    std::string test_lbl  = "../data/t10k-labels-idx1-ubyte";
    
    std::cout << "Loading datasets...\n";
    dataset::MNIST train_data, test_data;
    try {
        train_data = dataset::MNIST::load(train_img, train_lbl);
        test_data  = dataset::MNIST::load(test_img, test_lbl);
        std::cout << "Loaded " << train_data.num_samples << " training images.\n";
        std::cout << "Loaded " << test_data.num_samples << " testing images.\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to load dataset: " << e.what() << "\n";
        return 1;
    }

    // 2. Define the Model
    nn::Sequential model({
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 64),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(64, 10)
    });

    // 3. Initialize Optimizer (Learning Rate = 0.1)
    //optim::SGD opt(model.parameters(), 0.1f);
    optim::Adam opt(model.parameters(), 1e-3f);

    // 4. Hyperparameters
    size_t batch_size = 128;
    size_t epochs = 5;
    size_t train_batches = train_data.num_samples / batch_size;
    size_t test_batches  = test_data.num_samples / batch_size;

    std::cout << "Starting training (Epochs: " << epochs << ", Batch Size: " << batch_size << ")\n";
    std::cout << "───────────────────────────────────────────────────────────────────\n";

    // 5. The Main Loop
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // ─── A. TRAINING PHASE ───────────────────────────────────────────────
        float total_loss = 0.0f;
        int train_correct = 0;

        for (size_t b = 0; b < train_batches; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 784});
            std::vector<int> Y_batch(batch_size);

            float* x_ptr = X_batch->raw_ptr();
            const float* dataset_ptr = train_data.images.raw_ptr();
            size_t offset = b * batch_size;
            
            std::copy(dataset_ptr + offset * 784, dataset_ptr + (offset + batch_size) * 784, x_ptr);
            for (size_t i = 0; i < batch_size; ++i) Y_batch[i] = train_data.labels[offset + i];

            opt.zero_grad();
            auto logits = model.forward(X_batch);
            auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));
            
            total_loss += ops::nll_loss(log_probs, Y_batch);
            train_correct += static_cast<int>(ops::accuracy(logits, Y_batch) * batch_size);

            log_probs.backward();
            opt.step();
        }

        // ─── B. TESTING PHASE ────────────────────────────────────────────────
        int test_correct = 0;
        
        for (size_t b = 0; b < test_batches; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 784});
            std::vector<int> Y_batch(batch_size);

            float* x_ptr = X_batch->raw_ptr();
            const float* dataset_ptr = test_data.images.raw_ptr();
            size_t offset = b * batch_size;
            
            std::copy(dataset_ptr + offset * 784, dataset_ptr + (offset + batch_size) * 784, x_ptr);
            for (size_t i = 0; i < batch_size; ++i) Y_batch[i] = test_data.labels[offset + i];

            // NO ZERO GRAD, NO BACKWARD, NO OPT.STEP()! Just forward pass.
            auto logits = model.forward(X_batch);
            test_correct += static_cast<int>(ops::accuracy(logits, Y_batch) * batch_size);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // ─── C. REPORTING ────────────────────────────────────────────────────
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
    std::cout << "Run complete!\n";
    return 0;
}