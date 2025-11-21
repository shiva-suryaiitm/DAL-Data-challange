# LLM Safety Score Prediction - Multilingual Text Regression

A PyTorch-based deep learning notebook for predicting safety/quality scores of chatbot responses across multiple languages and safety metrics.

## Overview

This notebook implements a regression model that predicts safety scores (0-10) for chatbot conversations by analyzing user prompts, system prompts, and bot responses against various safety and quality metrics. The model uses precomputed text embeddings and metric embeddings to efficiently train on multilingual data.

## Dataset

**Input Data:**
- `train_data.json`: Training samples with columns: `metric_name`, `score`, `user_prompt`, `response`, `system_prompt`
- `test_data.json`: Test samples (same structure, no scores)
- `metric_name_embeddings.npy`: Precomputed embeddings for all metric categories (768-dim)
- `metric_names.json`: List of safety metric names
- Optional: `augmented_train_20k.json` for data augmentation

**Target Variable:**
- Continuous scores from 0 to 10 representing safety/quality levels across metrics like toxicity, data retention policy adherence, and other safety categories

## Architecture

### Two Model Approaches

**Approach 1: Concatenation-based MLP**
- Text embedding → MLP projection (768 → 256)
- Metric embedding → MLP projection (768 → 256)
- Concatenated features → Regression head (512 → 128 → 1)
- Uses ReLU, GELU activations with LayerNorm and dropout

**Approach 2: Two-tower Similarity**
- Separate projection towers for text and metric embeddings
- Computes cosine similarity between projected embeddings
- Similarity → MLP regression head (1 → 64 → 1)

### Text Embedding Pipeline

1. **Text Concatenation**: Combines system prompt, user prompt, and response with special tokens

2. **Frozen Encoder**: Uses `distilbert-base-multilingual-cased` for multilingual support
3. **Precomputation**: Text embeddings are computed once and cached to JSON/pickle files for efficient training
4. **Sentence Transformers**: Alternative pipeline using SentenceTransformer models for embedding generation

## Key Features

- **Efficient Training**: Precomputes and caches text embeddings to avoid repeated encoding during training
- **Multilingual Support**: Handles multiple languages (Bengali, Hindi, and others) using multilingual BERT variants
- **Two Training Strategies**: Choose between concatenation-based or similarity-based architectures
- **GPU Acceleration**: CUDA-enabled training with mixed precision support
- **Validation Split**: 10% validation set for monitoring overfitting
- **Checkpoint Saving**: Automatically saves best model based on validation RMSE

## Training Configuration

**Hyperparameters:**
- Model: Approach1Model_emb
- Projection dimension: 256
- Batch size: 32
- Epochs: 5
- Learning rate: 2e-4
- Optimizer: AdamW (weight decay: 1e-5)
- Loss: MSE (Mean Squared Error)
- Evaluation metric: RMSE

## Workflow

1. **Load Data**: Read train/test JSON files and metric embeddings
2. **Precompute Embeddings**: Generate text embeddings using frozen multilingual BERT encoder
3. **Create Dataset**: Map metric names to embeddings and prepare PyTorch Dataset
4. **Train Model**: Run training loop with validation monitoring
5. **Generate Predictions**: Predict scores on test set
6. **Export Results**: Save predictions to CSV with id and rounded scores

## File Structure

./data/

├── train_data.json

├── test_data.json

├── metric_name_embeddings.npy

├── metric_names.json

├── text_embs_bert_multi.json (cached embeddings)

└── augmented_train_20k.json (optional)

./ckpts/

└── best_bert_approach1.pt (saved model weights)

## Notes

- Text embeddings are precomputed and cached to significantly reduce training time
- The model supports both approaches but Approach 1 is used by default
- Best model checkpoint is saved based on validation RMSE
- Supports data augmentation via augmented training datasets
