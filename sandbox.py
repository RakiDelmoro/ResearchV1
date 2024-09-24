import random
import cupy as cp

def generate_pair_of_data(random_input):
    input_to_hidden_1_target_weights = cp.array([[-0.4534, 0.3253], [0.6575, 0.1234]])
    hidden_1_target_bias = cp.array([0.02341, -0.3463])

    hidden_1_to_hidden_2_target_weights = cp.array([[-0.6457, 0.0866], [0.5475, 0.4064]])
    hidden_2_target_bias = cp.array([-0.3435, -0.0456])

    hidden_2_to_output_target_weights = cp.array([[0.3467, 0.4572], [0.7897, 0.5675]])
    output_target_bias = cp.array([0.1123, 0.9763])

    while True:
        # generate_input = cp.array([[random.randint(0, 2), random.randint(0, 2)]])
        hidden_1_neurons = (cp.dot(random_input, input_to_hidden_1_target_weights)) + hidden_1_target_bias
        hidden_2_neurons = (cp.dot(hidden_1_neurons, hidden_1_to_hidden_2_target_weights)) + hidden_2_target_bias
        expected = (cp.dot(hidden_2_neurons, hidden_2_to_output_target_weights)) + output_target_bias

        yield expected

def neural_network():
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
        print(f"EPOCHS: {epochs+1}")
        
        generate_input = cp.array([[random.randint(0, 2), random.randint(0, 2)]])
        expected_data = next(iter(generate_pair_of_data(generate_input)))

        # Forward pass
        hidden_1_neurons = (cp.dot(generate_input, input_to_hidden_1_weights)) + hidden_1_bias
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
        hidden_2_to_output_weights -= 0.01 * cp.dot(hidden_2_neurons.transpose(), error_last_layer)
        output_bias -= 0.01 * cp.sum(error_last_layer, axis=0)
        hidden_1_to_hidden_2_weights -= 0.01 * cp.dot(hidden_1_neurons.transpose(), error_hidden_2)
        hidden_2_bias -= 0.01 * cp.sum(error_hidden_2, axis=0)
        input_to_hidden_1_weights -= 0.01 * cp.dot(generate_input.transpose(), error_hidden_1)
        hidden_1_bias -= 0.01 * cp.sum(error_hidden_1, axis=0)

        # Network stress
        network_stress = sum([error_last_layer, error_hidden_2, error_hidden_1])**2
        print(output_bias)
        print(f"Output neuron: {output_neurons} Expected: {expected_data}")
        print(f"Network stress: {cp.mean(network_stress)}")

        epochs += 1

# training_samples = generate_pair_of_data(1)
neural_network()
