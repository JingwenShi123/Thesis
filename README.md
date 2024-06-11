# Thesis
Enhanced Multimodal Emotion Recognition using GRU and Self-Attention Mechanisms: Techniques and Applications

# Multimodal Emotion Recognition Thesis

This repository contains the code and datasets associated with the Master's thesis "Multimodal Emotion Recognition Augmentation: Techniques and Applications" by Jingwen Shi. This work was conducted under the supervision of Assistant Professor Dr. Shekhar Nayak at the University of Groningen.

## Abstract

This thesis makes substantial contributions to the field of multimodal emotion recognition by developing and evaluating models that integrate audio, visual, and textual data. The proposed model architecture combines Gated Recurrent Units (GRUs) and self-attention mechanisms, significantly improving feature extraction and emotion recognition accuracy.

## Datasets

The following datasets were used in this study:

- **CMU-MOSI**: [CMU-MOSI GitHub](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- **CMU-MOSEI**: [CMU-MOSEI GitHub](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
- **CH-SIMS**: [CH-SIMS GitHub](https://aclanthology.org/2020.acl-main.343/)

Processed versions of these datasets are provided in PKL format and can be found in the `Dataset` directory.

## Experimental Setup

The experimental setup includes detailed descriptions of the datasets, model architectures, training procedures, and evaluation metrics. The main components are:

### Datasets

The datasets include video features extracted using pre-trained CNN models, audio features processed with the LibROSA library, and text features derived from BERT embeddings.

### Models

1. **GRUWithLinear Model**: Combines GRU with a linear layer for sequential data processing.
2. **MLP (Two-layered Perceptron) Model**: Uses fully connected layers with ReLU activation functions.
3. **SelfAttention Layer**: Implements self-attention for capturing significant features within sequences.
4. **EMIFusion Model**: Integrates information from multiple modalities using Low-Rank Tensor Fusion (LRTF).

### Training and Evaluation

- **Complexity.py**: Evaluates model performance and memory usage.
- **Performance.py**: Assesses effectiveness through metrics like F1 score, accuracy, and AUPRC.
- **Robustness.py**: Measures robustness to noise using relative and effective robustness metrics.

## Code Structure

- `getdata.py`: Data processing module for loading, preprocessing, and augmenting data.
- `models/`: Directory containing model definitions.
- `train.py`: Script for training models.
- `test.py`: Script for testing models.

## Usage

### Cloning the Repository

```bash
git clone https://github.com/JingwenShi123/Thesis.git
cd Thesis
