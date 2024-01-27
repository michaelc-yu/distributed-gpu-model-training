#
# Created by Michael Yu on 1/12/2024
#

import multiprocessing
import numpy as np
import time
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def predict(self, X):
        return self.weight * X + self.bias

    def update_params(self, gradient_weight, gradient_bias, learning_rate):
        self.weight -= learning_rate * gradient_weight
        self.bias -= learning_rate * gradient_bias


def train_parallel(process_id, model, x, y, batch_size, learning_rate, num_iterations):
    print("in train parallel")

    print(f"len x: {len(x)}") # 250
    print(f"len y: {len(y)}") # 250

    print(f"mini batch size: {batch_size}") # 50 - train 50 data then sync

    x_chunks = np.array_split(x, len(x) / batch_size)
    y_chunks = np.array_split(y, len(y) / batch_size)


    for i in range(len(x_chunks)): # loop 5 times
        for iteration in range(num_iterations):

            predictions = model.predict(x_chunks[i])

            gradient_weight = 2 * np.mean((predictions - y_chunks[i]) * x_chunks[i])
            gradient_bias = 2 * np.mean(predictions - y_chunks[i])

            model.update_params(gradient_weight, gradient_bias, learning_rate)

            print(f"Process {process_id}, Iteration {iteration + 1}, Weight: {model.get_weight()}, Bias: {model.get_bias()}")

        # now sync data between processes
        print(f"Model params before syncing. Weight: {model.get_weight()}, Bias: {model.get_bias()}")


    print(f"final weight: {model.get_weight()}, final bias: {model.get_bias()}")



# data parallelism using
        # bulk synchronous parallels (BSP) or
        # asynchronous parallel (ASP)
def distributed_training(model, num_processes, data_pairs, batch_size, learning_rate, num_iterations, synchronization_approach):
    print(f" we have: {data_pairs.shape[0]} data")
    print(data_pairs)
    X = data_pairs[:, 0]  # Extract the first column
    Y = data_pairs[:, 1]  # Extract the second column
    # print(f"X: {X}")
    # print(f"Y: {Y}")

    x_chunks = np.array_split(X, num_processes)
    y_chunks = np.array_split(Y, num_processes)
    # x and y will be split evenly into num_processes chunks

    processes = []

    for i in range(num_processes):
        process = multiprocessing.Process(target=train_parallel,
                                        args=(i, model, x_chunks[i], y_chunks[i], batch_size/num_processes, learning_rate, num_iterations))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    return


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate random data for the independent variable (X)
    X = 2 * np.random.rand(1000, 1)

    # Generate the corresponding dependent variable (y) with some noise (y = 3x + 4 + noise)
    Y = 3 * X + np.random.randn(1000, 1) + 4

    data_pairs = np.column_stack((X, Y))


    learning_rate = 0.01
    num_iterations = 10
    num_processes = 4 # number of gpus (or cpu cores) to distribute to
    batch_size = 200

    shared_model = SimpleLinearRegression()

    distributed_training(shared_model, num_processes, data_pairs, batch_size, learning_rate, num_iterations, "BSP")
