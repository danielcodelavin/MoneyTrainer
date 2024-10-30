import torch
import torch.nn as nn
import os

class StableTransformerModel(nn.Module):
    # The model class should mirror the one used in training for compatibility.
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(StableTransformerModel, self).__init__()
        # Model definition based on the training file
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim*2, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x[:, 0])

def load_checkpoint(filepath: str, model: nn.Module):
    """Loads the model checkpoint from the specified filepath."""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
    else:
        print(f"Checkpoint file not found at {filepath}")

def inference(input_tensor: torch.Tensor, model: nn.Module, device: torch.device):
    """Runs inference on the given input tensor."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    return output

def generate_input_tensor():
    """Placeholder for generating an input tensor for inference."""
    # Assume the input tensor generation code is implemented here
    input_tensor = torch.randn(1, 768)  # Example tensor; replace with actual generation logic
    return input_tensor

def main(checkpoint_path: str = "checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableTransformerModel(hidden_dim=768, num_layers=8, num_heads=12, dropout=0.1)
    model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model)
    
    # Generate input tensor and run inference
    input_tensor = generate_input_tensor()
    output = inference(input_tensor, model, device)
    
    print("Inference Output:", output)

if __name__ == "__main__":
    main(checkpoint_path="path/to/your/checkpoint.pt",ticker="AAPL")
