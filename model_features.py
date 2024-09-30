import torch

DEVICE = "cuda"
EPOCHS = 10
BATCH_SIZE = 2098
LEARNING_RATE = 6.0
MODEL_FEATURE_SIZE = [784, 2000, 10]
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
HIDDEN_LAYER_ACTIVATION = torch.nn.LeakyReLU()
OUTPUT_LAYER_ACTIVATION = torch.nn.Softmax(dim=-1)
