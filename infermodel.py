import torch
import os
import numpy as np

class SwiGLU(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, in_features)
        self.linear2 = torch.nn.Linear(in_features, in_features)
        self.beta = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.linear1(x) * torch.sigmoid(self.beta * self.linear2(x))

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, nhead)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        swiglu_output = self.swiglu(x)
        x = self.norm2(x + swiglu_output)
        return x

class TransformerRegression(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.input_projection = torch.nn.Linear(1, d_model)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.regression_head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x[-1]
        return self.regression_head(x).squeeze(-1)

def generate_data():
    """
    Placeholder function to generate a single 1D vector.
    Replace this with your actual data generation logic.
    """
    # For this example, we'll generate a random vector of length 100
    return torch.rand(100)

def load_model(checkpoint_path, d_model, nhead, num_layers):
    model = TransformerRegression(d_model, nhead, num_layers)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(model, input_data):
    with torch.no_grad():
        input_tensor = input_data.unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
    return output.item()

def main():
    # Model parameters (should match the parameters used during training)
    d_model = 768
    nhead = 8
    num_layers = 12

    # Path to the checkpoint file
    checkpoint_path = 'path/to/checkpoints/checkpoint_step_XXXXX.pt'

    # Load the model
    model = load_model(checkpoint_path, d_model, nhead, num_layers)

    # Generate data
    input_data = generate_data()

    # Make prediction
    prediction = predict(model, input_data)

    print(f"Predicted value: {prediction}")

if __name__ == "__main__":
    main()