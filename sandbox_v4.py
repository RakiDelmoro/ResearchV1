import cupy as cp
from generate_pair_data import generate_pair_of_data

def neural_network(network_architecture: list):
    forward_axons = [cp.random.randn(network_architecture[size], network_architecture[size+1]) for size in range(len(network_architecture)-1)]
    forward_dentrites = [cp.random.randn(network_architecture[size+1]) for size in range(len(network_architecture)-1)]
    
    reverse_network_architecture = network_architecture[::-1]
    backward_axons = [cp.random.randn(network_architecture[size], network_architecture[size+1]) for size in range(len(reverse_network_architecture)-1)]
    backward_dentrites = [cp.random.randn(network_architecture[size+1]) for size in range(len(reverse_network_architecture)-1)]

    def forward_in_neurons(neurons):
        neurons_activations = [neurons]
        for neurons_index in range(len(network_architecture)-1):
            neurons = cp.dot(neurons, forward_axons[neurons_index]) + forward_dentrites[neurons_index]
            neurons_activations.append(neurons)
        return neurons_activations

    def backward_in_neurons(neurons):
        neurons_activations = [neurons]
        for neurons_index in range(len(network_architecture)-1):
            neurons = cp.dot(neurons, backward_axons[neurons_index]) + backward_dentrites[neurons_index]
            neurons_activations.append(neurons)
        return neurons_activations

    def calculate_network_stress(forward_neurons_activations, backward_neurons_activations):
        neurons_losses = []
        for each_neurons_activations in range(len(network_architecture)):
            neurons_loss = forward_neurons_activations[each_neurons_activations] - backward_neurons_activations[-(each_neurons_activations+1)]
            neurons_losses.append(neurons_loss)
        return neurons_losses

    def forward_axons_and_dentrites_update(neurons_losses, forward_neurons_activations, learning_rate):
        for neurons_idx, (each_neurons_axons, each_neurons_dentrites) in enumerate(zip(forward_axons, forward_dentrites)):
            neurons_activation = forward_neurons_activations[neurons_idx]
            neurons_loss = neurons_losses[neurons_idx+1]

            each_neurons_axons -= learning_rate * cp.dot(neurons_activation.transpose(), neurons_loss)
            each_neurons_dentrites -= learning_rate * cp.sum(neurons_loss, axis=0)

    def backward_axons_and_dentrites_update(neurons_losses, backward_neurons_activations, learning_rate):
        for neurons_idx, (each_neurons_axons, each_neurons_dentrites) in enumerate(zip(backward_axons, backward_dentrites)):
            neurons_activation = backward_neurons_activations[neurons_idx]
            neurons_loss = neurons_losses[-(neurons_idx+2)]

            each_neurons_axons += learning_rate * cp.dot(neurons_activation.transpose(), neurons_loss)
            each_neurons_dentrites += learning_rate * cp.sum(neurons_loss, axis=0)

    def runner():
        while True:
            forward_pass_input_neurons, backward_pass_input_neurons = next(generate_pair_of_data(for_torch=False))

            forward_neurons_activations = forward_in_neurons(forward_pass_input_neurons)
            backward_neurons_activations = backward_in_neurons(backward_pass_input_neurons)

            neurons_loss = calculate_network_stress(forward_neurons_activations, backward_neurons_activations)
            forward_axons_and_dentrites_update(neurons_loss, forward_neurons_activations, learning_rate=0.01)
            backward_axons_and_dentrites_update(neurons_loss, backward_neurons_activations, learning_rate=0.1)

            print(cp.mean(sum(neurons_loss)**2))
            print(forward_neurons_activations[-1], backward_pass_input_neurons)
            print(backward_neurons_activations[-1], forward_pass_input_neurons)

    return runner

model = neural_network(network_architecture=[2, 2, 2, 2])
model()

