from torch import nn
import torch
import numpy as np



class SelfAttention(nn.Module):

    def __init__(self, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        # define attention params
        self.m = input_size
        self.output_size = output_size
        # define 3 vectors (K,Q,V)
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        # define output linear transformation layer
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)
        self.drop50 = nn.Dropout(0.5)

    def forward(self, x):
        # process the features through 3 layers k,q,v
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        # normalize the attention scores
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))
        attention_weights = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(attention_weights)
        # calculate final weighted values
        y = torch.matmul(weights, V)

        y = self.output_linear(y)

        return y



class VideoSummarizerNetwork(nn.Module):

    def __init__(self):
        super(VideoSummarizerNetwork, self).__init__()
        self.input_size = 1024
        self.hidden_size = 1024
        
        self.attention_layer = SelfAttention(input_size=self.input_size, output_size=self.input_size)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.norm_input = nn.LayerNorm(self.input_size)
        self.norm_hidden = nn.LayerNorm(self.hidden_size)

    def forward(self, video_features):
        feature_size = video_features.shape[2]  # Extract the feature size (third dimension)
    
        # Flatten the sequence into a batch of frames
        video_features = video_features.view(-1, feature_size)
        attention_output = self.attention_layer(video_features)
        residual_output = attention_output + video_features  # Residual connection (adding input to attention output)
        # Apply dropout and layer normalization on the residual output
        normalized_output = self.dropout(residual_output)
        normalized_output = self.norm_input(normalized_output)
        
        # Pass through the first fully connected layer, apply ReLU and dropout
        hidden_output = self.fc1(normalized_output)
        hidden_output = self.relu(hidden_output)
        hidden_output = self.dropout(hidden_output)
        hidden_output = self.norm_hidden(hidden_output)
        
        # Pass through the second fully connected layer to get the final output
        output = self.fc2(hidden_output)
        output = self.sigmoid(output).view(1, -1)

        return output

    def predict(self, sequence):
        # Make a prediction for a given sequence
        predictions = self(sequence)
        predictions = predictions[0].detach().cpu().numpy()
        return predictions
