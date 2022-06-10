
import torch
import torch.nn as nn
from transformers import HubertModel


MODEL = "facebook/hubert-base-ls960"


class HubertForAudioClassification(nn.Module):
    def __init__(self, base_model, adapter_hidden_size=64):
        super().__init__()

        self.hubert = base_model
        
        hidden_size = self.hubert.config.hidden_size

        self.adaptor = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),           
            nn.Dropout(0.1),            
            nn.Linear(adapter_hidden_size, hidden_size),
        )  
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),           
            nn.Dropout(0.1),            
            nn.Linear(adapter_hidden_size, 1),
        )
        
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()
    
    def forward(self, x):
        # x shape: (B,E)
        x = self.hubert(x).last_hidden_state
        
        x = self.adaptor(x)
        
        # pooling
        x, _ = x.max(dim=1)

        # Mutilayer perceptron
        out = self.classifier(x)
        # out shape: (B,1)

        # Remove last dimension
        # return out.squeeze(-1)
        # return shape: (B)
        return out

def build_hubert(adapter_hidden_size=64):

    base_model = HubertModel.from_pretrained(MODEL)
    model = HubertForAudioClassification(base_model, adapter_hidden_size=adapter_hidden_size)
    return model
