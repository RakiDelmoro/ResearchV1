import cupy as cp
from generate_pair_data import generate_pair_of_data

def neural_network(network_size: list):
    axons = [cp.random.randn(network_size[size], network_size[size+1]) for size in range(len(network_size)-1)]
    dentrites = [cp.random.randn(network_size[size+1]) for size in range(len(network_size)-1)]

    def forward_in_neurons(neurons):
        neurons_activations = [neurons]
        for neurons_index in range(len(network_size)-1):
            neurons = cp.dot(neurons, axons[neurons_index]) + dentrites[neurons_index]
            neurons_activations.append(neurons)

        return neurons_activations

    def backward_in_neurons(neurons):
        neurons_activations = [neurons]
        for neurons_index in range(len(network_size)-2):
            neurons = cp.dot(neurons, axons[-neurons_index+1].transpose()) + dentrites[-neurons_index+1]
            neurons_activations.append(neurons)

        return neurons_activations

    def calculate_network_stress(forward_pass_activations, backward_pass_activations):
        neurons_losses = []
        neuron_loss_and_previous_neurons = []
        for each_neurons_activation in range(len(network_size)-1):
            neurons_loss = forward_pass_activations[each_neurons_activation+1] - backward_pass_activations[-each_neurons_activation+1]
            previous_neurons = forward_pass_activations[each_neurons_activation]

            neurons_losses.append(neurons_loss)
            neuron_loss_and_previous_neurons.append((previous_neurons, neurons_loss))

        network_stress = sum(neurons_losses)**2
        return neuron_loss_and_previous_neurons, cp.mean(network_stress).item()

    def update_axons_and_dentrites(neurons_loss_and_previous_neurons, learning_rate=0.001):
        for neurons_idx, (each_neurons_axons, each_neurons_dentrites) in enumerate(zip(axons, dentrites)):
            previous_neurons = neurons_loss_and_previous_neurons[neurons_idx][0]
            neurons_loss = neurons_loss_and_previous_neurons[neurons_idx][1]

            each_neurons_axons -= learning_rate * cp.dot(previous_neurons.transpose(), neurons_loss)
            each_neurons_dentrites -= learning_rate * cp.sum(neurons_loss, axis=0)

    epochs = 0
    while True:
        print(f'EPOCHS: {epochs+1}')

        generated_input_neurons, generated_expected_output_neurons = next(generate_pair_of_data())
        forward_neurons_activations = forward_in_neurons(generated_input_neurons)
        backward_neurons_activations = backward_in_neurons(generated_expected_output_neurons)

        while True:
            neurons_loss_correspond_to_previous_neurons, network_stress = calculate_network_stress(forward_neurons_activations, backward_neurons_activations)
            update_axons_and_dentrites(neurons_loss_correspond_to_previous_neurons)
            forward_neurons_activations = forward_in_neurons(generated_input_neurons)
            backward_neurons_activations = backward_in_neurons(generated_expected_output_neurons)

            if round(network_stress, 6) == 0.0:
                break

        print(generated_expected_output_neurons, forward_neurons_activations[-1])
        epochs += 1

neural_network([2, 2, 2, 2])
