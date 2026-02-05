from vit import *
from llm import *
import torch.nn as nn

class VisualTransformer(nn.Module):
    def __init__(self, vit_config, llm_config, tokenizer):
        super().__init__()
        self.vit = get_vit(vit_config['img_size'], vit_config['patch_size'], vit_config['in_channels'], vit_config['d_model'], vit_config['num_heads'], vit_config['dropout'], llm_config['d_model'], vit_config['num_layers'] )
        self.transformer = llm_model(len(tokenizer), llm_config['seq_len'], llm_config['d_model'], llm_config['d_ff'], llm_config['num_heads'], llm_config['num_layers'], llm_config['dropout'])
    
    def forward(self, image, text_ids):
        vit_tokens = self.vit(image)

        text_tokens = self.transformer.inputEmbedding(text_ids)
        text_tokens = self.transformer.positionalEncoding(text_tokens)

        # Concatenate: [Batch, vit_tokens + text_tokens, d_model]
        combined_tokens = torch.cat([vit_tokens, text_tokens], dim=1)
        
        return self.transformer(combined_tokens, is_already_embedded=True)