import torch
import random
import cupy as cp

def generate_pair_of_data(for_torch=False):
    input_to_hidden_1_target_weights = cp.array([[-0.4534, 0.3253], [0.6575, 0.1234]], dtype=cp.float32)
    hidden_1_target_bias = cp.array([0.02341, -0.3463], dtype=cp.float32)
    hidden_1_to_hidden_2_target_weights = cp.array([[-0.6457, 0.0866], [0.5475, 0.4064]], dtype=cp.float32)
    hidden_2_target_bias = cp.array([-0.3435, -0.0456], dtype=cp.float32)
    hidden_2_to_output_target_weights = cp.array([[0.3467, 0.4572], [0.7897, 0.5675]], dtype=cp.float32)
    output_target_bias = cp.array([0.1123, 0.9763], dtype=cp.float32)

    # For torch
    input_to_hidden_1_target_weights_torch = torch.tensor([[-0.4534, 0.3253], [0.6575, 0.1234]], device="cuda", requires_grad=True, dtype=torch.float16)
    hidden_1_target_bias_torch = torch.tensor([0.02341, -0.3463], device="cuda", requires_grad=True, dtype=torch.float16)
    hidden_1_to_hidden_2_target_weights_torch = torch.tensor([[-0.6457, 0.0866], [0.5475, 0.4064]], device="cuda", requires_grad=True, dtype=torch.float16)
    hidden_2_target_bias_torch = torch.tensor([-0.3435, -0.0456], device="cuda", requires_grad=True, dtype=torch.float16)
    hidden_2_to_output_target_weights_torch = torch.tensor([[0.3467, 0.4572], [0.7897, 0.5675]], device="cuda", requires_grad=True, dtype=torch.float16)
    output_target_bias_torch = torch.tensor([0.1123, 0.9763], device="cuda", requires_grad=True, dtype=torch.float16)

    while True:
        if not for_torch:
            generated_input = cp.array([[random.randint(0, 2), random.randint(0, 2)]], dtype=cp.float32)
            hidden_1_neurons = cp.dot(generated_input, input_to_hidden_1_target_weights) + hidden_1_target_bias
            hidden_2_neurons = cp.dot(hidden_1_neurons, hidden_1_to_hidden_2_target_weights) + hidden_2_target_bias
            expected = cp.dot(hidden_2_neurons, hidden_2_to_output_target_weights) + output_target_bias
        else:
            generated_input = torch.tensor([[random.randint(0, 2), random.randint(0, 2)]], device="cuda", dtype=torch.float16)
            hidden_1_neurons = torch.nn.functional.linear(generated_input, input_to_hidden_1_target_weights_torch, hidden_1_target_bias_torch)
            hidden_2_neurons = torch.nn.functional.linear(hidden_1_neurons, hidden_1_to_hidden_2_target_weights_torch, hidden_2_target_bias_torch)
            expected = torch.nn.functional.linear(hidden_2_neurons, hidden_2_to_output_target_weights_torch, output_target_bias_torch)

        yield generated_input, expected