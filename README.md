# Distributed training with MPI

This is project is to implement the framework using MPI to transfer gradients among a local LAN cluster to train ML models, like ResNet.

## Requirements

1. [C++-17](http://www.cplusplus.com/doc/tutorial/introduction/) compatible compiler
2. [CMake](https://cmake.org/download/) (minimum version 3.14)
3. [LibTorch version >= 1.12.0](https://pytorch.org/cppdocs/installing.html)
4. MPI, OpenMPI or [MPICH2](https://mpitutorial.com/tutorials/installing-mpich2/) version >= 3.3.2

## Install, build and run

Download and unzip/install LibTorch/MPI, and then add the path to LibTorch to the CMakeLists.txt, letting the system know where your LibTorch source files are.

```bash
# Set torch path
set(CMAKE_PREFIX_PATH your_path_to_libtorch)
# e.g.
# set(CMAKE_PREFIX_PATH /root/CISC_830_programmingHomework/libtorch)
```

Build the program:

```bash
$ mkdir build && cd build
$ cmake ..
$ cmake --build ../build
$ make
```

These commands will build the program, write files to ./build and also download cifar10 dataset if it is not found locally. And the path to the dataset is at ./data/cifar10 by default, which will be used by the file system when training the ML model. If you want to use a different file path, change the code in the main.cpp file:

```c++
// File structure in cifar10:
// cifar10
//    - batches.meta.txt
//    - data_batch_1.bin
//    - data_batch_2.bin
//    - data_batch_3.bin
//    - data_batch_4.bin
//    - data_batch_5.bin
//    - data_batch_6.bin
//    - readme.html
//    - test_batch.bin
const std::string CIFAR_data_path = your_path_to_test_batch.bin;
const std::string CIFAR_data_path_root = your_path_to_data_batch_number_for_root.bin
const std::string CIFAR_data_path_worker_1 = your_path_to_data_batch_number_worker_1.bin
    
// e.g.
// File structure in CIFAR_data_path:
// cifar10
//    - test_batch.bin
const std::string CIFAR_data_path = "/root/CISC_830_programmingHomework/manully_update_grads/data/cifar10/";
// File structure in CIFAR_data_path_root:
// cifar10_root
//    - data_batch_1.bin
const std::string CIFAR_data_path_root = "/root/CISC_830_programmingHomework/manully_update_grads/data/cifar10_root/";
// File structure in CIFAR_data_path_worker_1:
// cifar10_worker_1
//    - data_batch_2.bin
const std::string CIFAR_data_path_worker_1 = "/root/CISC_830_programmingHomework/manully_update_grads/data/cifar10_worker_1/";
```

Finally, you need to change some parameters in the cifar10.cpp file if you want to use a different dataset setting different than mine, like kTrainSize, kTrainDataBatchFilesRoot, etc.

After that, you need to set up your MPI cluster shown in [this tutorial](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/) or just install MPI in your local machine for the minimal requirement, that is, testing with multiple nodes on a local machine. After that, you can run mpirun command and specify how many nodes you want to spawn in each node

```bash
$ mpirun -np nodes_you_want_to_spawn -hosts hosts_list ./deep-residual-network
# e.g running 2 nodes on two remote servers, master and worker1
# $ mpirun -np 2 -hosts master,worker1 ./deep-residual-network
# e.g running 2 nodes on local machine
# $ mpirun -np 2 -hosts localhost ./deep-residual-network
```

Here is the typical output with the example command of 2 nodes on local machine:

```c++
// $ mpirun -np 2 -hosts localhost ./deep-residual-network
// Running time here is in microseconds
Rank 0: Deep Residual Network

Rank 0: Training on CPU.
Rank 1: Deep Residual Network

Rank 1: Training on CPU.
Rank 1: Training...
Rank 0: Training...
Rank 0: Epoch [1/20], Trainset - Loss: Rank 1: Epoch [1/20], Trainset - Loss: 2.4236, Accuracy: 0.0994
Rank 1: Training finished!

Rank 1: Testing...
2.3920, Accuracy: 0.1533
Rank 0: Training finished!

Rank 0: Testing...
Rank 1: Testing finished!
Rank 1: Testset - Loss: 2.3464, Accuracy: 0.1038
Rank 1 : Running time: 276247745
Rank 0: Testing finished!
Rank 0: Testset - Loss: 2.3433, Accuracy: 0.1619
Rank 0 : Running time: 276313245
```

## Test and comparison

Tested the program with 2 nodes on one local machine compared to one local machine without MPI, i.e. centralized training. Testing environment: 

- RIT remote server, acharya-r1.main.ad.rit.edu.
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), each node has 10000 images, 20000 in total.
- ResNet
- CPU training

Please see the [results.xlsx](https://github.com/fengkeyleaf/Distributed_training_with_MPI/blob/main/results.xlsx) file for more details.

※ Ran 3 tests on each machine.
※ Planned to test with 2 remote servers, but still having an unknown problem after trying different MPI versions and using one local machine and one RIT remote server.

## Q & A

Q: What if I get [an error](https://stackoverflow.com/questions/34143265/undefined-reference-to-symbol-pthread-createglibc-2-2-5), like "/usr/bin/ld: CMakeFiles/deep-residual-network.dir/src/main.cpp.o: undefined reference to symbol 'pthread_create@@GLIBC_2.2.5'"

A: Add the following CMake command to the CMakeLists.txt file:

```cmake
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
```

## Acknowledges

1. [pytorch-cpp](https://github.com/prabhuomkar/pytorch-cpp)
2. [MPI Tutorial](https://mpitutorial.com/tutorials/)
3. [pytorch_cpp](https://github.com/koba-jon/pytorch_cpp)

