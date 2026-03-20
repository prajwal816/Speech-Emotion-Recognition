import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class CNNLSTM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config["model"]
        
        # CNN Feature Extractor
        cnn_filters = self.config["cnn_filters"]
        layers = []
        in_ch = 1
        for out_ch in cnn_filters:
            layers.append(CNNBlock(in_ch, out_ch))
            in_ch = out_ch
            
        self.cnn = nn.Sequential(*layers)
        
        # Calculate feature size after CNN
        # n_mels / 2^len(cnn_filters)
        self.freq_dim = config["data"]["n_mels"] // (2 ** len(cnn_filters))
        self.lstm_in_size = cnn_filters[-1] * self.freq_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_size,
            hidden_size=self.config["lstm_hidden_size"],
            num_layers=self.config["lstm_num_layers"],
            batch_first=True,
            dropout=self.config["dropout"] if self.config["lstm_num_layers"] > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(self.config["dropout"])
        self.fc = nn.Linear(self.config["lstm_hidden_size"], self.config["num_classes"])
        
    def forward(self, x):
        # x shape: (batch_size, channels, n_mels, time)
        # Check if single channel, if missing add it
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.cnn(x)
        # x shape: (batch_size, channels, freq, time)
        
        # Prepare for LSTM: (batch_size, time, channels * freq)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, channels, freq = x.size()
        x = x.reshape(batch_size, time_steps, channels * freq)
        
        # LSTM encoding
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Aggregation over time steps (mean pooling)
        x = torch.mean(lstm_out, dim=1) 
        
        # Final classification
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
