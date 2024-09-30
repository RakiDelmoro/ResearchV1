import torch
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_features import BATCH_SIZE, EPOCHS, LEARNING_RATE, HIDDEN_LAYER_ACTIVATION, OUTPUT_LAYER_ACTIVATION, LOSS_FUNCTION, MODEL_FEATURE_SIZE, DEVICE

class MlpNetwork(torch.nn.Module):
    def __init__(self, model_feature_size: list, layer_activation_function: torch.nn, output_activation_function: torch.nn,
                 loss_function: torch.nn, learning_rate, device: str):
        super().__init__()
        self.device = device
        self.loss_func = loss_function
        self.model_feature_size = model_feature_size
        self.activation_function = layer_activation_function
        self.output_activation_function = output_activation_function
        
        self.layers = torch.nn.ModuleList([Linear(model_feature_size[each_feature_size], model_feature_size[each_feature_size+1], device=device)
                                           for each_feature_size in range(len(model_feature_size)-1)])

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    
    def forward(self, batch_data):
        previous_layer_output = batch_data.flatten(1, -1)
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers)-1:
                previous_layer_output = self.activation_function(layer(previous_layer_output))
            else:
                previous_layer_output = self.output_activation_function(layer(previous_layer_output))

        return previous_layer_output

    def training_loop(self, training_dataloader):
        per_batch_losses = []
        for input_batch, expected_batch in training_dataloader:
            input_batch = input_batch.to(self.device)
            expected_batch = expected_batch.to(self.device)
            model_prediction = self.forward(input_batch)
            loss = self.loss_func(model_prediction, expected_batch)
            
            per_batch_losses.append(loss.item())
            
            # Update model parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        batched_average_loss = sum(per_batch_losses) / len(per_batch_losses)
        print(f"Training result: Loss -> {batched_average_loss}")

    def test_loop(self, validation_dataloader):
        test_loss = 0
        correct_prediction = 0
        # Don't need to apply gradient since we don't train it
        with torch.no_grad():
            for input_batch, expected_batch in validation_dataloader:
                input_batch = input_batch.to(self.device)
                expected_batch = expected_batch.to(self.device)
 
                model_prediction_in_probability = self.forward(input_batch)
                model_prediction_in_index = model_prediction_in_probability.argmax(dim=-1, keepdim=True)
                
                test_loss += torch.nn.CrossEntropyLoss(reduction='sum').forward(model_prediction_in_probability, expected_batch)
                correct_prediction += model_prediction_in_index.eq(expected_batch.view_as(model_prediction_in_index)).sum().item()
        
        test_loss /= len(validation_dataloader.dataset)
        print(f"Test result: Loss -> {test_loss} Model accuracy -> {correct_prediction}/{len(validation_dataloader.dataset)} percentage -> {100. * correct_prediction / len(validation_dataloader.dataset)}%")

def runner():
    transform = transforms.Compose([transforms.ToTensor()])

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=transform)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=transform)

    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mlp_model = MlpNetwork(model_feature_size=MODEL_FEATURE_SIZE, layer_activation_function=HIDDEN_LAYER_ACTIVATION,
                           output_activation_function=OUTPUT_LAYER_ACTIVATION, learning_rate=LEARNING_RATE, loss_function=LOSS_FUNCTION, device=DEVICE)

    for epoch in range(EPOCHS):
        print(F'EPOCHS: {epoch+1}')
        mlp_model.training_loop(training_dataloader)
        mlp_model.test_loop(validation_dataloader)

runner()