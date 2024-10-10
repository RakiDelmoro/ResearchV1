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

        yield generated_input, expected

def neural_network():
    # Input to first hidden neurons weights and bias
    input_to_hidden_1_weights = cp.random.randn(2, 2)
    hidden_1_bias = cp.random.randn(2)
    # First hidden neurons to second hidden neurons weights and bias
    hidden_1_to_hidden_2_weights = cp.random.randn(2, 2)
    hidden_2_bias = cp.random.randn(2)
    # second hidden neurons to output neurons weights and bias
    hidden_2_to_output_weights = cp.random.randn(2, 2)
    output_bias = cp.random.randn(2)

    epochs = 0
    while True:
        print(f"EPOCHS: {epochs+1}")
        generated_input, generated_expected = next(generate_pair_of_data())

        # forward pass
        hidden_1_forward_neurons = cp.dot(generated_input, input_to_hidden_1_weights) + hidden_1_bias
        hidden_2_forward_neurons = cp.dot(hidden_1_forward_neurons, hidden_1_to_hidden_2_weights) + hidden_2_bias
        output_forward_neurons = cp.dot(hidden_2_forward_neurons, hidden_2_to_output_weights) + output_bias

        # backward pass (Inverse function of forward pass)
        hidden_2_neurons_inverse = cp.dot(generated_expected - output_bias, cp.linalg.pinv(hidden_2_to_output_weights))
        hidden_1_neurons_inverse = cp.dot(hidden_2_neurons_inverse - hidden_2_bias, cp.linalg.pinv(hidden_1_to_hidden_2_weights))

        # We train each layer independently we start with the first hidden layer forward
        last_layer_error = output_forward_neurons - generated_expected
        hidden_2_layer_error = hidden_2_forward_neurons - hidden_2_neurons_inverse
        hidden_1_layer_error = hidden_1_forward_neurons - hidden_1_neurons_inverse

        # Network stress
        network_stress = sum([last_layer_error, hidden_2_layer_error, hidden_1_layer_error])**2
        print(f"Input to hidden 1 weights: {hidden_2_to_output_weights}")
        print(f"Network stress: {cp.mean(network_stress)}")

        # Update model parameters
        hidden_2_to_output_weights -= 0.001 * cp.dot(hidden_2_forward_neurons.transpose(), last_layer_error)
        output_bias -= 0.001 * cp.sum(last_layer_error, axis=0)
        
        hidden_1_to_hidden_2_weights -= 0.001 * cp.dot(hidden_1_forward_neurons.transpose(), hidden_2_layer_error)
        hidden_2_bias -= 0.001 * cp.sum(hidden_2_layer_error, axis=0)
        
        input_to_hidden_1_weights -= 0.001 * cp.dot(generated_input.transpose(), hidden_1_layer_error) 
        hidden_1_bias -= 0.001 * cp.sum(hidden_1_layer_error, axis=0)

neural_network()
