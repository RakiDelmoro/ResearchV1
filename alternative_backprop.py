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

    # Expected to hidden 2 neurons
    expected_to_hidden_2_weights = cp.random.randn(2, 2)
    hidden_2_bias_backward = cp.random.randn(2)
    # Hidden 2 neurons to hidden 1 neurons
    hidden_2_to_hidden_1_weights = cp.random.randn(2, 2)
    hidden_1_bias_backward = cp.random.randn(2)
    # hidden 1 neurons to input data
    hidden_1_to_input_data_weights = cp.random.randn(2, 2)
    input_bias_backward = cp.random.randn(2)

    epochs = 0
    while True:
        # Forward pass
        hidden_1_neurons = (cp.dot(input_data, input_to_hidden_1_weights)) + hidden_1_bias
        hidden_2_neurons = (cp.dot(hidden_1_neurons, hidden_1_to_hidden_2_weights)) + hidden_2_bias
        output_neurons = (cp.dot(hidden_2_neurons, hidden_2_to_output_weights)) + output_bias
        # backward pass
        predicted_output_neurons = (cp.dot(output_neurons, expected_to_hidden_2_weights)) + hidden_2_bias_backward
        predicted_hidden_2_neurons = (cp.dot(predicted_output_neurons, hidden_2_to_hidden_1_weights)) + hidden_1_bias_backward
        predicted_hidden_1_neurons = (cp.dot(predicted_hidden_2_neurons, hidden_1_to_input_data_weights)) + input_bias_backward
        # Errors for each layer
        error_for_output_neurons = predicted_output_neurons - expected_data
        error_for_hidden_2_neurons = predicted_hidden_2_neurons - hidden_2_neurons
        error_for_hidden_1_neurons = predicted_hidden_1_neurons - hidden_1_neurons

        input_to_hidden_1_weights += 0.001 * cp.dot(input_data.transpose(), error_for_hidden_1_neurons)
        hidden_1_to_hidden_2_weights += 0.001 * cp.dot(hidden_1_neurons.transpose(), error_for_hidden_2_neurons)
        hidden_2_to_output_weights += 0.001 * cp.dot(hidden_2_neurons.transpose(), error_for_output_neurons)
        hidden_1_bias += 0.001 * cp.sum(error_for_hidden_1_neurons, axis=0)
        hidden_2_bias += 0.001 * cp.sum(error_for_hidden_2_neurons, axis=0)
        output_bias += 0.001 * cp.sum(error_for_output_neurons, axis=0)

        expected_to_hidden_2_weights -= 0.001 * cp.dot(output_neurons.transpose(), error_for_output_neurons)
        hidden_2_to_hidden_1_weights -= 0.001 * cp.dot(predicted_output_neurons.transpose(), error_for_hidden_2_neurons)
        hidden_1_to_input_data_weights -= 0.001 * cp.dot(predicted_hidden_2_neurons.transpose(), error_for_hidden_1_neurons)
        hidden_2_bias_backward -= 0.001 * cp.sum(error_for_output_neurons, axis=0)
        hidden_1_bias_backward -= 0.001 * cp.sum(error_for_hidden_2_neurons, axis=0)
        input_bias_backward -= 0.001 * cp.sum(error_for_hidden_1_neurons, axis=0)
        # Network stress
        network_stress = sum([error_for_output_neurons, error_for_hidden_2_neurons, error_for_hidden_1_neurons])**2
        print(f"EPOCHS: {epochs+1}")
        print(f"output neuron {predicted_output_neurons} expected: {expected_data}")
        print(f"Network stress: {cp.mean(network_stress)}")

        epochs += 1

x = cp.random.randn(1, 2)
y = cp.random.randint(0, 2, (1, 2))
neural_network(x, y)
