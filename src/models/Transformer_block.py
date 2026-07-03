import torch
import torch.nn as nn

class Transformer_Block(nn.Module):
    """
    Bloc d'encodage Transformer standard.
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        batch_first: bool = True,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant de l'encodeur Transformer.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée. 
                              Shape: (batch_size, seq_len, d_model) si batch_first=True.
                              Shape: (seq_len, batch_size, d_model) si batch_first=False.
                              
        Returns:
            torch.Tensor: Représentation encodée de même dimension que l'entrée.
        """
        return self.transformer_encoder(x)