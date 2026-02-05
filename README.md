## Project Title: Custom-Transformer-VLM

A Modular Vision Transformer and Decoder-Only LLM Implementation from Scratch

## Overview

This repository contains a from-scratch implementation of a Vision-Language Model (VLM) architecture. It demonstrates the structural integration of a Vision Transformer (ViT) and 
a Causal Decoder LLM through a custom projection bridge. The project focuses on the tensor-level logic of multimodal data fusion. 

## Key Components

- vit.py (Vision Backbone): Implements a patch-based Linear Projection (1024 x 1024 input to 16 x 16 patches), a learnable extra embedding (CLS-style), and a stack of N Transformer Encoder layers.
- llm.py (Language Backbone): A Decoder-only Transformer featuring multi-head self-attention with causal masking, absolute positional encodings, and a customizable layer depth.
- vlm.py (Integration Layer): The "bridge" architecture. It coordinates the concatenation of visual and textual tokens and manages the multimodal causal mask to ensure correct information flow during cross-modal inference.
- config.py: Centralised hyperparameter management for both the ViT (patch size, d_model, heads) and the LLM (vocab size, sequence length, d_ff).

## Technical Highlights

- Weight Initialisation: All modules utilise Xavier Uniform initialisation to maintain gradient stability across deep stacks (N layers).
- Masking Strategy: Implements a bitwise AND-based causal mask that handles both temporal causality and padding simultaneously.
- Residual Paths: Pre-layer normalisation with dropout is implemented across all Transformer blocks to prevent vanishing gradients during training simulations.
