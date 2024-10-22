import random
import cupy as cp

def generate_pair_of_data():
    input_to_hidden_1_target_weights = cp.array([[-0.4534, 0.3253], [0.6575, 0.1234]])
    hidden_1_target_bias = cp.array([0.02341, -0.3463])

    hidden_1_to_hidden_2_target_weights = cp.array([[-0.6457, 0.0866], [0.5475, 0.4064]])
    hidden_2_target_bias = cp.array([-0.3435, -0.0456])

    hidden_2_to_output_target_weights = cp.array([[0.3467, 0.4572], [0.7897, 0.5675]])
    output_target_bias = cp.array([0.1123, 0.9763])

    while True:
        generated_input = cp.array([[random.randint(0, 2), random.randint(0, 2)]])
        hidden_1_neurons = cp.dot(generated_input, input_to_hidden_1_target_weights) + hidden_1_target_bias
        hidden_2_neurons = cp.dot(hidden_1_neurons, hidden_1_to_hidden_2_target_weights) + hidden_2_target_bias
        expected = cp.dot(hidden_2_neurons, hidden_2_to_output_target_weights) + output_target_bias

        yield expected

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
        print(f"EPOCHS: {epochs+1}")
        
        # Forward pass
        hidden_1_neurons = cp.dot(input_data, input_to_hidden_1_weights) + hidden_1_bias
        hidden_2_neurons = cp.dot(hidden_1_neurons, hidden_1_to_hidden_2_weights)  + hidden_2_bias
        output_neurons = cp.dot(hidden_2_neurons, hidden_2_to_output_weights) + output_bias

        hidden_2_neurons_target = cp.dot(expected_data, hidden_2_to_output_weights.transpose()) + hidden_2_bias
        hidden_1_neurons_target = cp.dot(hidden_2_neurons_target, hidden_1_to_hidden_2_weights.transpose()) + hidden_1_bias

        # # Each layer error
        error_last_layer = output_neurons - expected_data
        error_hidden_2 = hidden_2_neurons - hidden_2_neurons_target
        error_hidden_1 = hidden_1_neurons - hidden_1_neurons_target

        hidden_2_to_output_weights -= 0.01 * cp.dot(hidden_2_neurons.transpose(), error_last_layer)
        output_bias -= 0.01 * cp.sum(error_last_layer, axis=0)
        hidden_1_to_hidden_2_weights -= 0.01 * cp.dot(hidden_1_neurons.transpose(), error_hidden_2)
        hidden_2_bias -= 0.01 * cp.sum(error_hidden_2, axis=0)
        input_to_hidden_1_weights -= 0.01 * cp.dot(input_data.transpose(), error_hidden_1)
        hidden_1_bias -= 0.01 * cp.sum(error_hidden_1, axis=0)

        # # Network stress
        network_stress = sum([error_last_layer, error_hidden_2, error_hidden_1])**2
        print(f"Output neuron: {output_neurons} Expected: {expected_data}")
        print(f"Network stress: {cp.mean(network_stress)}")

        epochs += 1

x = cp.array([[1.0, 0.0]])
y = cp.array([[1.0, 1.0]])

# training_samples = generate_pair_of_data(1)
neural_network(x, y)
