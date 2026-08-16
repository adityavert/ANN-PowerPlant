# ANN Regression — Power Plant Energy Output Prediction
 
An Artificial Neural Network built with PyTorch to predict the **net hourly electrical energy output (PE)** of a combined cycle power plant based on ambient environmental conditions.
 
---
 
## Dataset Features
 
| Feature | Description            | Unit  |
|---------|------------------------|-------|
| AT      | Ambient Temperature    | °C    |
| V       | Exhaust Vacuum         | cm Hg |
| AP      | Ambient Pressure       | mbar  |
| RH      | Relative Humidity      | %     |
| **PE**  | Net Energy Output *(Target)* | MW |
 
---
 
## Model Architecture
 
```
Input (4)  →  Dense(6, ReLU)  →  Dense(6, ReLU)  →  Output (1)
```
 
| Hyperparameter   | Value       |
|------------------|-------------|
| Optimizer        | Adam        |
| Loss Function    | MSE         |
| Epochs           | 100         |
| Batch Size       | 32          |
| Train/Test Split | 80% / 20%   |
 
---
 
## Pipeline
 
```
Raw CSV  →  Null Check  →  Train-Test Split  →  StandardScaler
         →  PyTorch Tensors  →  DataLoader  →  Training  →  Evaluation
```
 
---
 
## Evaluation
 
- **MSE** on train and test sets
- **R² Score** on the test set
- **Loss Curve** — Training vs. Validation loss across 100 epochs
- Best model weights saved as `best_model.pt`
---
 
## Key Concepts Covered
 
- Custom ANN using `nn.Module` and `nn.Sequential`
- Mini-batch training with `TensorDataset` and `DataLoader`
- Feature scaling with `StandardScaler`
- Validation loop using `torch.no_grad()`
- Model checkpointing with `torch.save` / `torch.load`
- Regression evaluation with MSE and R² Score
