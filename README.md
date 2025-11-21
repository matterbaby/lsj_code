# MAISNet: A Multi-Species Integrated Graph Neural Network for Acetylcholinesterase Inhibitor Screening
MAISNet is an intelligent screening framework for acetylcholinesterase inhibitors (AChEIs) based on deep learning. It integrates data from six species to address the inefficiency of traditional AChEI screening methods. The model encodes inhibitors as SMILES-derived molecular graphs and AChE structures as residue contact maps, extracting multi-scale features via SAGE and GAT networks, and fusing them through bidirectional cross-attention to predict binding affinity. MAISNet outperforms state-of-the-art models in accuracy and robustness, providing an efficient tool for accelerating Alzheimer's disease therapeutic discovery.

## Key Features
- **Multi-species Integration**: Utilizes AChE data from six species (Homo sapiens, Electrophorus electricus, Mus musculus, Bos taurus, Torpedo californica, Rattus norvegicus) to enhance generalization.
- **Advanced Feature Extraction**: Combines SAGE (molecular features) and GAT (protein features) with multi-scale graph convolution to capture local and global structural patterns.
- **Bidirectional Cross-Attention**: Enables deep fusion of protein-ligand interactions for accurate binding affinity prediction.
- **Practical Application**: Successfully identifies high-affinity AChEI candidates through virtual screening and molecular docking.

## Datasets
- **Dataset A**: Collected from BindingDB (accessed 3 March 2024), containing 19,845 inhibitor molecules from six species, split into training (80%), validation (10%), and test (10%) sets.
- **Dataset B**: External validation set from ChEMBL (accessed 7 August 2025), with 1,997 inhibitor molecules targeting the same six species.
- **Screening Dataset**: 51,220 unvalidated small molecules from the ZINC database (accessed 1 July 2025) for candidate screening.

## Operating Environment
- **Programming Language**: Python 3.10
- **Deep Learning Framework**: PyTorch 2.1.0
- **Operating System**: Ubuntu 22.04
- **Hardware Configuration**: NVIDIA RTX 4090 GPU (24 GB VRAM), AMD EPYC 9654 96-core CPU (2.40 GHz)
- **Dependencies**: Pillow, PyTorch, NumPy, Scikit-learn, RDKit (for molecular processing)

## Model Training & Evaluation
### Training Settings
- Epochs: 100
- Batch Size: 32
- Optimizer: Adam (initial learning rate = 0.001, weight decay = 1e-5)
- Learning Rate Schedule: Linear warm-up (first 10 epochs) + ReduceLROnPlateau (decay factor = 0.7, patience = 5)
- Dropout Rate: 0.2 (for GAT and SAGE layers)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC), AUC-ROC


## Code Files
- `classification.py`: MAISNet core structure (feature extraction, cross-attention, MLP predictor).
- `data_preprocessing.py`: Data loading, SMILES encoding, residue contact map construction.
- `main.py`: Model training, validation, and performance evaluation.
- `O42275`
- `P04058`
- `P21836`
- `P22303`
- `P23795`

## Usage
1. **Data Preparation**: Process inhibitor SMILES and AChE structures using `data_preprocessing.py`.
2. **Model Training**: Run `train_evaluation.py` to train MAISNet and validate on test sets.
3. **Inhibitor Screening**: Use `inhibitor_screening.py` to predict potential AChEIs and perform molecular docking.
4. **Visualization**: Generate species summary images or model performance plots with provided scripts.
