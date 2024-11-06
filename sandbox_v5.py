import cupy as cp
from features import GREEN, RED, RESET
from generate_pair_data import generate_pair_of_data

def neural_network(network_architecture: list):
    # Initializae axons and dentrites for forward pass and backward pass
    forward_axons = [cp.random.randn(network_architecture[size_idx], network_architecture[size_idx+1]) for size_idx in range(len(network_architecture)-1)]
    forward_dentrites = [cp.zeros(network_architecture[size_idx+1]) for size_idx in range(len(network_architecture)-1)]
    # Backward pass axons and dentrites same as forward axons and dentrties except for layer_neurons 2 to output neurons
    reverse_network_architecture = network_architecture[::-1]
    backward_axons = [forward_axons[-(size_idx+1)].transpose() if size_idx != len(reverse_network_architecture)-2 else cp.random.randn(reverse_network_architecture[size_idx], reverse_network_architecture[size_idx+1])
                      for size_idx in range(len(reverse_network_architecture)-1)]
    backward_dentrites = [forward_dentrites[-(size_idx+2)] if size_idx != len(reverse_network_architecture)-2 else cp.zeros(reverse_network_architecture[size_idx+1])
                          for size_idx in range(len(reverse_network_architecture)-1)]

    def forward_in_neurons(neurons):
        neurons_activations = [neurons]
        for layer_idx in range(len(network_architecture)-1):
            neurons = cp.dot(neurons, forward_axons[layer_idx]) + forward_dentrites[layer_idx]
            neurons_activations.append(neurons)
        return neurons_activations

    def backward_in_neurons(neurons):
        neurons_activations = [neurons]
        for layer_idx in range(len(network_architecture)-1):
            neurons = cp.dot(neurons, backward_axons[layer_idx]) + backward_dentrites[layer_idx]
            neurons_activations.append(neurons)
        return neurons_activations

    def calculate_network_stress(forward_activations, backward_activations):
        forward_pass_stress = []
        backward_pass_stress = []
        tuple_of_forward_neurond_activation_and_stress = []
        tuple_of_backward_neurons_activation_and_stress = []
        for activation_idx in range(len(network_architecture)-1):
            forward_activation = forward_activations[-(activation_idx+2)]
            forward_neurons_stress = forward_activations[-(activation_idx+1)] - backward_activations[activation_idx]
            backward_activation = backward_activations[-(activation_idx+2)]
            backward_neurons_stress = backward_activations[-(activation_idx+1)] - forward_activations[activation_idx]
            
            tuple_of_forward_neurond_activation_and_stress.append((forward_activation, forward_neurons_stress))
            tuple_of_backward_neurons_activation_and_stress.append((backward_activation, backward_neurons_stress))
            forward_pass_stress.append(cp.mean(forward_neurons_stress))
            backward_pass_stress.append(cp.mean(backward_neurons_stress))
        return tuple_of_forward_neurond_activation_and_stress, tuple_of_backward_neurons_activation_and_stress, forward_pass_stress, backward_pass_stress

    def update_axons_and_dentrites(neurons_and_stress, axons_to_update: list, dentrites_to_update: list, for_backward_pass: bool=False):
        # Update from output connection to input connection
        for layer_idx in range(len(network_architecture)-1):
            neurons_activation = neurons_and_stress[layer_idx][0]
            neurons_stress = neurons_and_stress[layer_idx][1]

            axons_to_update[-(layer_idx+1)] -= 0.1 * cp.dot(neurons_activation.transpose(), neurons_stress)
            dentrites_to_update[-(layer_idx+1)] -= 0.1 * cp.sum(neurons_stress, axis=0)

    def runner():
        epochs = 0
        while True:
            print(f'EPOCHS: {epochs+1}')
            forward_input, backward_input = next(generate_pair_of_data())
            forward_pass_activations = forward_in_neurons(forward_input)
            backward_pass_activations = backward_in_neurons(backward_input)
            forward_neurons_and_stress, backward_neurons_and_stress, forward_layers_stress, backward_layers_stress = calculate_network_stress(forward_pass_activations, backward_pass_activations)
            print(f'Forward Layers Stress: {cp.mean(sum(forward_layers_stress)**2)}')
            print(f'Backward Layers Stress: {cp.mean(sum(backward_layers_stress)**2)}')
            print(f'{GREEN}Forward Pass Prediction{RESET}: {forward_pass_activations[-1]},  {RED}Backward Pass Input{RESET}: {backward_pass_activations[0]}')
            print(f'{GREEN}Backward Pass Prediction{RESET}: {backward_pass_activations[-1]}, {RED}Forward Pass Input{RESET}: {forward_pass_activations[0]}')
            update_axons_and_dentrites(forward_neurons_and_stress, forward_axons, forward_dentrites)
            update_axons_and_dentrites(backward_neurons_and_stress, backward_axons, backward_dentrites, True)
            epochs += 1

    return runner

neural_network(network_architecture=[2, 2, 2, 2])()
