import cupy as cp

def neural_network(input_data, expected_data, learning_rate):
    # Input to first hidden neurons weights and bias
    input_forward_pass_to_hidden_1_weights = cp.random.randn(2, 2)
    hidden_1_bias_forward_pass = cp.random.randn(2)
    # First hidden neurons to second hidden neurons weights and bias
    hidden_1_to_hidden_2_forward_pass_weights = cp.random.randn(2, 2)
    hidden_2_bias_forward_pass = cp.random.randn(2)
    # Second hidden neurons to output hidden neurons weights and bias
    hidden_2_to_last_layer_forward_pass_weights = cp.random.randn(2, 2)
    last_layer_bias_forward_pass = cp.random.randn(2)

    # Expected to hidden 2 neurons
    input_backward_pass_to_hidden_2_weights = cp.random.randn(2, 2)
    hidden_2_bias_backward_pass = cp.random.randn(2)
    # Hidden 2 neurons to hidden 1 neurons
    hidden_2_to_hidden_1_backward_pass_weights = cp.random.randn(2, 2)
    hidden_1_bias_backward_pass = cp.random.randn(2)
    # hidden 1 neurons to input data
    hidden_1_to_last_layer_backward_pass_weights = cp.random.randn(2, 2)
    last_layer_bias_backward_pass = cp.random.randn(2)

    epochs = 0
    while True:
        # Forward pass
        hidden_1_neurons_forward_pass = (cp.dot(input_data, input_forward_pass_to_hidden_1_weights)) + hidden_1_bias_forward_pass
        hidden_2_neurons_forward_pass = (cp.dot(hidden_1_neurons_forward_pass, hidden_1_to_hidden_2_forward_pass_weights)) + hidden_2_bias_forward_pass
        last_layer_forward_pass = (cp.dot(hidden_2_neurons_forward_pass, hidden_2_to_last_layer_forward_pass_weights))  + last_layer_bias_forward_pass
        # backward pass
        hidden_2_neurons_backward_pass = (cp.dot(expected_data, input_backward_pass_to_hidden_2_weights)) + hidden_2_bias_backward_pass
        hidden_1_neurons_backward_pass = (cp.dot(hidden_2_neurons_backward_pass, hidden_2_to_hidden_1_backward_pass_weights)) + hidden_1_bias_backward_pass
        last_layer_backward_pass = (cp.dot(hidden_1_neurons_backward_pass, hidden_1_to_last_layer_backward_pass_weights)) + last_layer_bias_backward_pass
        # Errors for each layer
        error_for_last_layer_forward_pass = last_layer_forward_pass - expected_data
        error_for_hidden_2_neurons = hidden_2_neurons_backward_pass - hidden_2_neurons_forward_pass
        error_for_hidden_1_neurons = hidden_1_neurons_backward_pass - hidden_1_neurons_forward_pass
        error_for_last_layer_backward_pass = last_layer_backward_pass - input_data

        # Network stress is sum of forward pass layers and backward pass layers difference squared
        network_stress = sum([error_for_last_layer_forward_pass, error_for_hidden_2_neurons, error_for_hidden_1_neurons, error_for_last_layer_backward_pass])**2

        # For forward pass parameters update 
        # TODO: update input_forward_pass to hidden_1_neurons_forward_pass weights and bias so that the activation will be same as hidden_1_neurons_backward_pass
        input_forward_pass_to_hidden_1_weights = input_forward_pass_to_hidden_1_weights + (learning_rate * error_for_hidden_1_neurons * input_data)
        hidden_1_bias_forward_pass = hidden_1_bias_forward_pass + (learning_rate * error_for_hidden_1_neurons).ravel()
        # TODO: update hidden_1_neurons_forward_pass to hidden_2_neurons_forward_pass weights and bias so that the activation will be same as hidden_2_neurons_backward_pass
        hidden_1_to_hidden_2_forward_pass_weights = hidden_1_to_hidden_2_forward_pass_weights + (learning_rate * error_for_hidden_2_neurons * hidden_1_neurons_forward_pass)
        hidden_2_bias_forward_pass = hidden_2_bias_forward_pass + (learning_rate * error_for_hidden_2_neurons).ravel()
        # TODO: update hidden_2_neurons_forward_pass to last_layer_forward_pass weights and bias so that the activation will be same as expected_data
        hidden_2_to_last_layer_forward_pass_weights = hidden_2_to_last_layer_forward_pass_weights + (learning_rate * error_for_last_layer_forward_pass * hidden_2_neurons_forward_pass)
        last_layer_bias_forward_pass = last_layer_bias_backward_pass + (learning_rate * error_for_last_layer_forward_pass ).ravel()

        # For backward parameters update
        # # TODO: update last_layer_forward_pass to hidden_2_neurons_backward_pass weights and bias so that the activation will be same as hidden_2_neurons_forward_pass
        # hidden_2_to_hidden_1_backward_pass_weights = hidden_2_to_hidden_1_backward_pass_weights + learning_rate * error_for_hidden_2_neurons * expected_data
        # hidden_2_bias_backward_pass = hidden_2_bias_backward_pass + (learning_rate * error_for_hidden_2_neurons).ravel()
        # # # TODO: update hidden_2_neurons_backward_pass to hidden_1_neurons_backward_pass weights and bias so that the activation will be same as hidden_1_neurons_forward_pass
        # hidden_2_to_hidden_1_backward_pass_weights = hidden_2_to_hidden_1_backward_pass_weights + learning_rate * error_for_hidden_1_neurons * hidden_2_neurons_backward_pass
        # hidden_1_bias_backward_pass = hidden_1_bias_backward_pass + (learning_rate * error_for_hidden_1_neurons).ravel()
        # # # TODO: update hidden_1_neurons_backward_pass to last_layer_backward_pass weights and bias so that the activation will be same as input_data
        # hidden_1_to_last_layer_backward_pass_weights = hidden_1_to_last_layer_backward_pass_weights + learning_rate * error_for_last_layer_backward_pass * hidden_1_neurons_backward_pass
        # last_layer_bias_backward_pass = last_layer_bias_backward_pass + (learning_rate * error_for_last_layer_backward_pass).ravel()

        print(f"EPOCHS: {epochs+1}")
        # print(f"input neuron: {last_layer_backward_pass} input: {input_data}")
        print(f"output neuron {last_layer_forward_pass} expected: {expected_data}")
        print(f"Network stress: {cp.mean(network_stress)}")
        epochs += 1

x = cp.random.randn(1, 2)
y = cp.random.randint(0, 2, (1, 2))
neural_network(x, y, 0.001)
