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
        hidden_1_neurons_forward_pass = (cp.dot(input_data, input_to_hidden_1_weights)) + hidden_1_bias
        hidden_2_neurons_forward_pass = (cp.dot(hidden_1_neurons_forward_pass, hidden_1_to_hidden_2_weights)) + hidden_2_bias
        last_layer_forward_pass = (cp.dot(hidden_2_neurons_forward_pass, hidden_2_to_output_weights)) + output_bias
        # backward pass
        hidden_2_neurons_backward_pass = (cp.dot(last_layer_forward_pass, expected_to_hidden_2_weights)) + hidden_2_bias_backward
        hidden_2_neurons_backward_pass = (cp.dot(hidden_2_neurons_backward_pass, hidden_2_to_hidden_1_weights)) + hidden_1_bias_backward
        last_layer_backward_pass = (cp.dot(hidden_2_neurons_backward_pass, hidden_1_to_input_data_weights)) + input_bias_backward

        # Errors for each layer
        error_for_output_neurons = last_layer_forward_pass - expected_data
        error_for_hidden_2_neurons = hidden_2_neurons_backward_pass - hidden_2_neurons_forward_pass
        error_for_hidden_1_neurons = hidden_2_neurons_backward_pass - hidden_1_neurons_forward_pass
        error_for_input_neurons = last_layer_backward_pass - input_data

        # Network stress is sum of forward pass layers and backward pass layers difference squared
        network_stress = sum([error_for_output_neurons, error_for_hidden_2_neurons, error_for_hidden_1_neurons, error_for_input_neurons])**2
        
        # For forward pass parameters update 
        # TODO: update input_forward_pass to hidden_1_neurons_forward_pass weights and bias so that the activation will be same as hidden_1_neurons_backward_pass
        # TODO: update hidden_1_neurons_forward_pass to hidden_2_neurons_forward_pass weights and bias so that the activation will be same as hidden_2_neurons_backward_pass
        # TODO: update hidden_2_neurons_forward_pass to last_layer_forward_pass weights and bias so that the activation will be same as expected_data

        # For backward parameters update
        # TODO: update last_layer_forward_pass to hidden_2_neurons_backward_pass weights and bias so that the activation will be same as hidden_2_neurons_forward_pass
        # TODO: update hidden_2_neurons_backward_pass to hidden_1_neurons_backward_pass weights and bias so that the activation will be same as hidden_1_neurons_forward_pass
        # TODO: update hidden_1_neurons_backward_pass to last_layer_backward_pass weights and bias so that the activation will be same as input_data
        
        print(f"EPOCHS: {epochs+1}")
        print(f"input neuron: {last_layer_backward_pass} input: {input_data}")
        print(f"output neuron {last_layer_forward_pass} expected: {expected_data}")
        print(f"Network stress: {cp.mean(network_stress)}")

        epochs += 1

x = cp.random.randn(1, 2)
y = cp.random.randint(0, 2, (1, 2))
neural_network(x, y)
