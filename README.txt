SN1 vs SN2 Reaction Mechanism Predictor

Predict whether a chemical reaction will follow an SN1 or SN2 mechanism based on substrate, nucleophile, solvent, and leaving group.

Files in this project:
- SN1-SN2-Reaction-Predictor.py — Main Python script to run predictions.
- SN1_SN2_Mechanisms.csv — Dataset used to train the model.
- SN1.png & SN2.png — Illustrations of the reaction mechanisms.
- requirements.txt — Python packages required to run the code.

Quick Setup:
1. Install Python 3.10+ if not installed.
2. Install dependencies:
   pip install -r requirements.txt
3. Keep all files in the same folder to ensure paths work correctly.

How to Run:
- Run the predictor:
   python SN1-SN2-Reaction-Predictor.py
- If the model is not yet trained, it will train automatically using SN1_SN2_Mechanisms.csv.
- The program will prompt you for:
   • Substrate degree (methyl, primary, secondary, tertiary)
   • Special environment (none, allylic, benzylic, neopentyl)
   • Nucleophile strength (strong, moderate, weak)
   • Solvent (protic, aprotic)
   • Leaving group (h2o, br, cl, i, ots, oms, otf, f, tso)
- After entering these values, it will:
   1. Display the predicted mechanism (SN1 or SN2)
   2. Show an explanation for the prediction
   3. Display the corresponding reaction image

Chemistry: 
-This model predicts which mechanism the reaction is likely to take depending on the reaction conditions, leaving group, and substrate. Data was pulled from text sources and prediction is based off common chemistry trends. However, since chemistry has exceptions to some rules and trends be advised some predictions may not be accurate. 
-Reaction model can not make predictions on leaving groups that are not in allowed_values.
-Substrate is ...
-Special environment is a functional group that may affect sterics, leaving group BLANK, and nucleophile strength
-Nucleophile strength is slightly arbitrary as they are generally not placed into strength categories.


Notes:
- File locations: Make sure SN1_SN2_Mechanisms.csv and images are in the same folder as the script.
- Dependencies: pandas, scikit-learn, joblib, IPython.display.
- If packages are missing:
   pip install pandas scikit-learn joblib ipython
- Mac users: If your folder names have spaces (like "Python ML"), wrap paths in quotes when using the terminal.

Optional:
Include screenshots of the program running with sample inputs to showcase functionality without requiring them to run code.
