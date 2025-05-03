# Domain Decomposed Physics-Informed Neural Networks  
# for Coupled Diffusion and Deformation in Bilayer Electrode Materials

---

## Project Overview
This project develops a domain-decomposed Physics-Informed Neural Network (PINN) framework to simulate the coupled diffusion and mechanical deformation in bilayer electrode materials, commonly used in lithium-ion batteries.  

To address material heterogeneities and discontinuities at interfaces, separate subnetworks are constructed for modeling the concentration and displacement fields within each subdomain, with enforced interface continuity conditions.

Two subnetwork optimization strategies are systematically investigated:
- **X-type (Simultaneous training)**
- **AX-type (Alternating training)**

Both **strong-form** and **energy-form** formulations are considered.

Three representative numerical cases validate the proposed framework's accuracy and robustness.  
Comparisons against high-fidelity finite element method (FEM) solutions show:
- Strong-form PINNs achieve excellent accuracy for simple coupled scenarios
- Energy-form approaches, especially combined with AX-type optimization, demonstrate superior stability and robustness in handling highly nonlinear behaviors and dynamic interface evolution.
---

## Method Summary
This project implements four fundamental variants:
- **X-PINN**: Strong-form + simultaneous optimization
- **X-DEM**: Energy-form + simultaneous optimization
- **AX-PINN**: Strong-form + alternating optimization
- **AX-DEM**: Energy-form + alternating optimization

---

## Physical Scenarios Covered
- Small deformation
- Softening modulus
- Moving interface

---

## Environment Requirements
- Python >= 3.10
- PyTorch 1.11 (GPU version recommended, or compatible versions like 2.0+)
- numpy < 2.0 (e.g., numpy==1.26.4)
- matplotlib (with LaTeX rendering support)
- pandas
- scipy
- (optional) LaTeX environment (TeXLive, MacTeX, MikTeX, etc.)

---

## Installation
We recommend using a conda environment:

```bash
conda create -n bilayer-dis python=3.10
conda activate bilayer-dis
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install matplotlib numpy==1.26.4 pyyaml pandas scipy
```
## How to Use

1. **Training from Scratch**:
    - Run any of the scripts below to train a model from scratch:
      ```
      python small_deformation/X-PINN.py
      python small_deformation/X-DEM.py
      python small_deformation/AX-PINN.py
      python small_deformation/AX-DEM.py
      # Similarly under softening_modulus/ or moving_interface_complex/
      ```
2. **Using Pretrained Models**:
    - Pretrained models are saved under `/model/` directories.
    - You can load them directly for evaluation or prediction.

3. **Plotting and Evaluation**:
    - Use scripts in the `plot/` folder to:
        - Visualize predicted concentration and displacement fields.
        - Compare against FEM reference solutions.
        - Calculate L2 errors.
    - Note: LaTeX installation is required to properly render figure labels.

4. **Output Files**:
    - Trained models (`.pth` files) will be saved.
    - Loss values will be printed during training.
    - Generated plots will be saved under the `figure/` folder.

---

### Project Structure
```
Bilayer-DIS/
    ├── FEM/                     # Finite Element Method (FEM) benchmark results
    ├── figure/                  # Figures generated from predictions
    ├── model/                   # Saved trained model files
    ├── moving_interface_complex/ # Moving interface related models and scripts
    ├── small_deformation/       # Small deformation related models and scripts
    ├── softening_modulus/       # Softening modulus related models and scripts
    ├── plot/                    # Plotting scripts for visualization and error analysis
    ├── utility/                 # Utility functions (network, loss functions, etc.)
    └── config.yaml              # Physical and training configuration file
```

### Notes
- LaTeX installation is required if you wish to generate figures with LaTeX-rendered labels
- Ensure numpy version < 2.0 for compatibility with PyTorch 1.11
- Using GPU is strongly recommended for significantly faster training

License
-------

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and redistribute the code, provided that proper credit is given to the original authors.