import cupy as cp

def neural_network(input_data, expected_data):
    # Input to first hidden neurons weights and bias
    input_to_hidden_1_weights = cp.random.randn(2, 2)
    hidden_1_bias = cp.random.randn(2)
    # First hidden neurons to second hidden neurons weights and bias
    hidden_1_to_hidden_2_weights = cp.random.randn(2, 2)
    hidden_2_bias = cp.random.randn(2)
    # Second hidden neurons to output hidden neurons weights and bias
    hidden_2_to_output_weights = cp.random.randn(2, 2)
    output_bias = cp.random.randn(2)

    epochs = 0
    while True:
        # Forward pass
        hidden_1_neurons = (cp.dot(input_data, input_to_hidden_1_weights)) + hidden_1_bias
        hidden_2_neurons = (cp.dot(hidden_1_neurons, hidden_1_to_hidden_2_weights)) + hidden_2_bias
        output_neurons = (cp.dot(hidden_2_neurons, hidden_2_to_output_weights)) + output_bias

        # Weight use same as forward pass but transposed
        expected_to_hidden_2_weights = hidden_2_to_output_weights.transpose()
        hidden_2_hidden_1_weights = hidden_1_to_hidden_2_weights.transpose()

        # backwad pass (It's just calculating each layers target so that we can calculate each layer loss)
        hidden_2_neurons_target = (cp.dot(expected_data, expected_to_hidden_2_weights)) + hidden_2_bias
        hidden_1_neurons_target = (cp.dot(hidden_2_neurons_target, hidden_2_hidden_1_weights)) + hidden_1_bias
        
        # Each layer error
        error_last_layer = output_neurons - expected_data
        error_hidden_2 = hidden_2_neurons - hidden_2_neurons_target
        error_hidden_1 = hidden_1_neurons - hidden_1_neurons_target

        # update parameters
        hidden_2_to_output_weights -= 0.001 * cp.dot(hidden_2_neurons.transpose(), error_last_layer)
        output_bias -= 0.001 * cp.sum(error_last_layer, axis=0)
        hidden_1_to_hidden_2_weights -= 0.001 * cp.dot(hidden_1_neurons.transpose(), error_hidden_2)
        hidden_2_bias -= 0.001 * cp.sum(error_hidden_2, axis=0)
        input_to_hidden_1_weights -= 0.001 * cp.dot(input_data.transpose(), error_hidden_1)
        hidden_1_bias -= 0.001 * cp.sum(error_hidden_1, axis=0)

        # Network stress
        network_stress = sum([error_last_layer, error_hidden_2, error_hidden_1])**2

        print(f"EPOCHS: {epochs+1}")
        print(f"output neuron {output_neurons} expected: {expected_data}")
        print(f"Network stress: {cp.mean(network_stress)}")

        epochs += 1

x = cp.random.randn(1, 2)
y = cp.random.randint(0, 2, (1, 2))
neural_network(x, y)
