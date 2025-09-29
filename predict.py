# predict.py
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# 1. Model jo aapne train kiya wo load karo
model_data = joblib.load("rf_fp_model_pro.pkl")
model = model_data['model']
threshold = model_data['threshold']
fp_radius = model_data['fp_radius']
fp_nbits = model_data['fp_nbits']

# 2. Helper functions
def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def mol_to_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)

def fp_to_numpy(fp):
    arr = np.zeros((fp_nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# 3. Woh molecule jiski aapko prediction chahiye
test_smiles = "CC1(CC2=C(CO1)SC(=C2C(=O)O)NC(=S)NC(=O)C3=CC=CC=C3)C"



# 4. SMILES ko fingerprint mein badlo
test_mol = mol_from_smiles(test_smiles)
if test_mol is None:
    print("Yeh SMILES sahi nahi hai.")
    exit()

test_fp = mol_to_fp(test_mol)
test_fp_array = fp_to_numpy(test_fp).reshape(1, -1)

# 5. Prediction karo
proba = model.predict_proba(test_fp_array)
class_names = model.classes_
# 'active' class ka index dhundho
active_index = list(class_names).index('active')
active_probability = proba[0][active_index]

# 6. Threshold use karke final prediction banao
prediction = "active" if active_probability >= threshold else "inactive"

# 7. Result ko print karo
print(f"\nPrediction for: {test_smiles}")
print(f"Predicted probability of being 'active': {active_probability:.4f} ({active_probability*100:.2f}%)")
print(f"Optimal threshold from training: {threshold:.4f}")
print(f"Final prediction: {prediction.upper()}")