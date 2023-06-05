import torch
from torchvision.models.vision_transformer import vit_b_16, vit_b_32
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = vit_b_16()

        self.dropout = backbone.encoder.dropout

        # 2,3,4,3
        self.layer1 = backbone.encoder.layers[:2]
        self.layer2 = backbone.encoder.layers[2:5] 
        self.layer3 = backbone.encoder.layers[5:9] 
        self.layer4 = backbone.encoder.layers[9:] 

        self.image_size = backbone.image_size
        self.patch_size = backbone.patch_size
        self.hidden_dim = backbone.hidden_dim
        self.mlp_dim = backbone.mlp_dim
        self.attention_dropout = backbone.attention_dropout
        self.num_classes = backbone.num_classes
        self.representation_size = backbone.representation_size
        self.norm_layer = backbone.norm_layer
        self.conv_proj = backbone.conv_proj
        self.class_token = backbone.class_token
        self.pos_embedding = backbone.encoder.pos_embedding
        pass

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1) # 到时候改成prompt就行
        x = torch.cat([batch_class_token, x], dim=1)


        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        x = x + self.pos_embedding
        x = self.dropout(x)
        x1 = self.layer1(x) # 3 197 768 --> 3 * 14 * 14 * 768
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.ln(x4)


        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

if __name__ == '__main__':
    data = torch.rand([3,3,224,224])
    a = Model()
    a(data)