#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>

#include <mpi.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "resnet.h"
#include "cifar10.h"
#include "transform.h"

// https://en.cppreference.com/w/cpp/filesystem/current_path
namespace fs = std::filesystem;

using resnet::ResNet;
using resnet::ResidualBlock;
using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

// reference materials:
// https://github.com/prabhuomkar/pytorch-cpp/tree/master/tutorials/intermediate/deep_residual_network

// TODO: Directly use Tensor to transfer via MPI.
void root_work( torch::Tensor& param, int world_size, double lr ) {
    torch::NoGradGuard no_grad;

    // Total number of gradients in this parameter tensor.
    int n = param.grad().numel();
    assert( n >= 0 );

    // Flatten the gradient tensor to transfer via MPI.
    auto s = param.grad().sizes(); // Original shape.
    torch::Tensor f = param.grad().view( n ); // Flatten to 1D array.
    assert( n == f.numel() );

    // Tensor to array.
//    std::cout << "Working on G" << std::endl;
    double G[ n ] = { 0 }; // Initialize all values in G to 0.
    for ( size_t i = 0; i < n; i++ ) {
        G[ i ] = f[ i ].item<double>();
    }

    // Total number of gradients from individual worker, including the root worker.
    size_t t = n * world_size;
    double R[ t ] = { 0 }; // Array to gather gradients.
//    std::cout << "Gathering" << std::endl;
    // Gather results from each worker node.
    MPI_Gather(
            G, n, MPI_DOUBLE,
            R, n, MPI_DOUBLE,
            0, MPI_COMM_WORLD
    );

    // Reset G b/c we want to reuse it.
    // And it already has the gradients of the root,
    // but we will receive them from MPI API.
    for ( size_t i = 0; i < n; i++ ) {
        G[ i ] = 0;
    }

//    std::cout << "Aggregating" << std::endl;
    // Aggregate results from all workers including the root worker.
    for ( size_t i = 0; i < t; i++ ) {
        G[ i % n ] += R[ i ];
    }

    // Average gradients.
    for ( size_t i = 0; i < n; i++ ) {
        G[ i ] /= world_size;
    }

//    std::cout << "Broadcasting" << std::endl;
    // Broadcast results to each worker node.
    MPI_Bcast( G, n, MPI_DOUBLE, 0, MPI_COMM_WORLD );

    // Array to tensor.
    for ( size_t i = 0; i < n; i++ ) {
        f[ i ] = G[ i ];
    }

//    std::cout << "Updating params" << std::endl;
    // Reshape the flattened gradient tensor to its origin shape.
    f = f.view( s );
    // Apply the learning rate and update parameter value using gradients
    param -= f * lr;
}

void worker_work( torch::Tensor& param, double lr ) {
    torch::NoGradGuard no_grad;

    int n = param.grad().numel();
    assert( n >= 0 );

    auto s = param.grad().sizes();
    torch::Tensor f = param.grad().view( n );

    double G[ n ];
    for ( size_t i = 0; i < n; i++ ) {
        G[ i ] = f[ i ].item<double>();
    }

    MPI_Gather(
            G, n, MPI_DOUBLE,
            nullptr, 0, MPI_DOUBLE,
            0, MPI_COMM_WORLD
    );

    // Broadcast results to each worker node.
    MPI_Bcast( G, n, MPI_DOUBLE, 0, MPI_COMM_WORLD );

    // Array to tensor.
    for ( size_t i = 0; i < n; i++ ) {
        f[ i ] = G[ i ];
    }

    // Reshape the flattened gradient tensor to its origin shape.
    f = f.view( s ); // Apply the learning rate and update parameter value using gradients
    param -= f * lr;
}

void training( int world_rank, int world_size ) {
    std::cout << "Rank " << world_rank << ": Deep Residual Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device( cuda_available ? torch::kCUDA : torch::kCPU );
    std::cout << "Rank " << world_rank  << ( cuda_available ? ": CUDA available. Training on GPU." : ": Training on CPU." ) << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 20;
    const double learning_rate = 0.001;
    const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
    const double learning_rate_decay_factor = 1.0 / 3.0;

    const std::string CIFAR_data_path = "/root/CISC_830_programmingHomework/manually_update_grads/data/cifar10/";
    const std::string CIFAR_data_path_root = "/root/CISC_830_programmingHomework/manually_update_grads/data/cifar10_root/";
    const std::string CIFAR_data_path_worker_1 = "/root/CISC_830_programmingHomework/manually_update_grads/data/cifar10_worker_1/";

    // CIFAR10 custom dataset
    auto train_dataset = CIFAR10(
            world_rank == 0 ? CIFAR_data_path_root : CIFAR_data_path_worker_1,
            world_rank
    )
    .map( ConstantPad( 4 ) )
    .map( RandomHorizontalFlip() )
    .map( RandomCrop( { 32, 32 } ) )
    .map( torch::data::transforms::Stack<>() );

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = CIFAR10( CIFAR_data_path, world_rank, CIFAR10::Mode::kTest )
            .map( torch::data::transforms::Stack<>() );

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move( train_dataset ), batch_size );

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move( test_dataset ), batch_size );

    // Model
    std::array<int64_t, 3> layers { 2, 2, 2 };
    ResNet<ResidualBlock> model( layers, num_classes );
    model->to( device );

    // Optimizer
    torch::optim::Adam optimizer( model->parameters(), torch::optim::AdamOptions( learning_rate ) );

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision( 4 );

    auto current_learning_rate = learning_rate;

    std::cout << "Rank " << world_rank << ": Training...\n";

    // Train the model
    for ( size_t epoch = 0; epoch != num_epochs; ++epoch ) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        int c = 0;
        for ( auto &batch: *train_loader ) {
            // Transfer images and target labels to device
            auto data = batch.data.to( device );
            auto target = batch.target.to( device );

            // Forward pass
            auto output = model->forward( data );

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy( output, target );

            // Update running loss
            running_loss += loss.item<double>() * data.size( 0 );

            // Calculate prediction
            auto prediction = output.argmax( 1 );

            // Update number of correctly classified samples
            num_correct += prediction.eq( target ).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();

//             optimizer.step();
            // https://stackoverflow.com/questions/65920683/what-is-the-libtorch-equivalent-to-pytorchs-torch-no-grad
            // Manually update parameters using gradients
            for ( auto &param: model->parameters() ) {
                if ( world_rank == 0 ) {
                    root_work( param, world_size, current_learning_rate );
                }
                else {
                    worker_work( param, current_learning_rate );
                }

//                break;
            }

//            if ( c++ == 3 ) break;
        }

        // Decay learning rate
        if ( ( epoch + 1 ) % learning_rate_decay_frequency == 0 ) {
            current_learning_rate *= learning_rate_decay_factor;
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups().front()
                    .options()).lr( current_learning_rate );
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Rank " << world_rank << ": Epoch [" << ( epoch + 1 ) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << '\n';

        break;
    }

    std::cout << "Rank " << world_rank << ": Training finished!\n\n";
    std::cout << "Rank " << world_rank << ": Testing...\n";

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for ( const auto &batch: *test_loader ) {
        auto data = batch.data.to( device );
        auto target = batch.target.to( device );

        auto output = model->forward( data );

        auto loss = torch::nn::functional::cross_entropy( output, target );
        running_loss += loss.item<double>() * data.size( 0 );

        auto prediction = output.argmax( 1 );
        num_correct += prediction.eq( target ).sum().item<int64_t>();
    }

    std::cout << "Rank " << world_rank << ": Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Rank " << world_rank << ": Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

int main() {
    MPI_Init( NULL, NULL );

//    std::cout << "Current path is " << fs::current_path() << '\n'; // (1)

    int world_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    int world_size;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    auto start = std::chrono::high_resolution_clock::now();
    training( world_rank, world_size );
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Rank " << world_rank << " : Running time: " <<
    std::chrono::duration_cast<std::chrono::microseconds>( stop - start ).count()
    << std::endl;

    MPI_Finalize();
}
