# SN1 vs SN2 Reaction Mechanism Predictor

This model predicts whether a nucleophilic substitution reaction is more likely to follow the **SN1** or **SN2** pathway based on reaction conditions.  
The prediction considers **substrate type**, **special environment**, **nucleophile strength**, **solvent**, and **leaving group**.  

The dataset was compiled from textbook references, and the model follows **commonly taught organic chemistry trends**.  
However, chemistry has exceptions to its rules—so some predictions may not be accurate in unusual or borderline cases.  

### Files in this project:
- SN1-SN2-Reaction-Predictor.py — Main Python script to run predictions.
- SN1_SN2_Mechanisms.csv — Dataset used to train the model.
- SN1.png & SN2.png — Illustrations of the reaction mechanisms.
- SN1vsSN2 (1).ipynb - Interactive Jupyter Notebook
- requirements.txt — Python packages required to run the code.

### Quick Setup:
1. Install Python 3.10+ if not installed.
2. Install dependencies:
   pip install -r requirements.txt

## How to Run:
- Run the predictor:
   python SN1-SN2-Reaction-Predictor.py
  - To run on Jupyter notebook, download Anaconda-Navigator
  - cd into the directory where the notebook is stored.
  - Run jupyter notebook
- If the model is not yet trained, it will train automatically using SN1_SN2_Mechanisms.csv.
- The program will prompt you for: 
   - Substrate degree (methyl, primary, secondary, tertiary)
   - Special environment (none, allylic, benzylic, neopentyl)
   - Nucleophile strength (strong, moderate, weak)
   - Solvent (protic, aprotic)
   - Leaving group (h2o, br, cl, i, ots, oms, otf, f, tso)
- After entering these values, it will:
   1. Display the predicted mechanism (SN1 or SN2)
   2. Show an explanation for the prediction
   3. Display the corresponding reaction image

### Chemistry: 
- **Leaving groups** must be one of the options listed in `allowed_values`. If leaving group is not listed, the model cannot make a prediction.  
- **Substrate** refers to the carbon atom bonded to the leaving group, described by its degree of substitution:  
- **Special environment** refers to structural features or functional groups that may alter reactivity by:  
  - Affecting sterics (crowding around the reactive site)  
  - Stabilizing or destabilizing a leaving group    
  Examples: allylic, benzylic, neopentyl. 
- **Protic vs Aprotic Solvents** 
  - **Protic solvents** tend to decrease SN2 reaction rates due to solvation of the nucleophile.  
  - **Aprotic solvents** tend to increase SN2 reaction rates by solvating the neighboring cation, leaving the nucleophile more reactive. 
- **Nucleophile strength** categories (`strong`, `moderate`, `weak`) are a simplification. In reality, nucleophile strength is continuous and depends on multiple factors (charge, polarizability, solvent effects, etc.).  



### Notes:
- File locations: Make sure SN1_SN2_Mechanisms.csv and images are in the same folder as the script.
- Dependencies: pandas, scikit-learn, joblib, IPython.display.
- If packages are missing:
   pip install pandas scikit-learn joblib ipython

