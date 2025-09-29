import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from io import BytesIO
from PIL import Image
import base64

# st.set_page_config(page_title="Compound Bioactivity Predictor", page_icon="🧪", layout="wide")

# # ----------------------
# # Helpers
# # ----------------------
# @st.cache_resource
# def load_model(path="rf_fp_model_pro.pkl"):
#     try:
#         data = joblib.load(path)
#         return data
#     except Exception as e:
#         st.error(f"Failed to load model file: {e}")
#         return None

# @st.cache_data
# def mol_from_smiles(smiles: str):
#     try:
#         return Chem.MolFromSmiles(smiles)
#     except:
#         return None

# @st.cache_data
# def canonical_smiles(smiles: str):
#     m = mol_from_smiles(smiles)
#     return Chem.MolToSmiles(m, canonical=True) if m else None

# @st.cache_data
# def mol_to_fp_numpy(mol, radius=2, nbits=2048):
#     if mol is None:
#         return None
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
#     arr = np.zeros((nbits,), dtype=np.uint8)
#     DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# @st.cache_data
# def tanimoto_similarity(fp1, fp2):
#     return DataStructs.TanimotoSimilarity(fp1, fp2)

# def render_molecule_image(smiles, size=(350, 200)):
#     m = mol_from_smiles(smiles)
#     if not m:
#         return None
#     img = Draw.MolToImage(m, size=size)
#     bio = BytesIO()
#     img.save(bio, format="PNG")
#     return bio.getvalue()

# # ----------------------
# # Load model
# # ----------------------
# with st.spinner("Loading model..."):
#     model_data = load_model()

# if not model_data:
#     st.stop()

# model = model_data.get("model")
# default_threshold = float(model_data.get("threshold", 0.5))
# fp_radius = int(model_data.get("fp_radius", 2))
# fp_nbits = int(model_data.get("fp_nbits", 2048))
# active_smiles_list = model_data.get("active_canonical_smiles", [])

# # Precompute RDKit bitvectors for actives for fast similarity
# active_mols = [mol_from_smiles(smi) for smi in active_smiles_list]
# active_fps_rd = [AllChem.GetMorganFingerprint(m, fp_radius) if m else None for m in active_mols]

# # ----------------------
# # Sidebar: Settings
# # ----------------------
# st.sidebar.header("Settings")
# threshold = st.sidebar.slider("Decision threshold (probability for 'active')", 0.0, 1.0, float(default_threshold), 0.01)
# show_similar = st.sidebar.checkbox("Show top similar actives", value=True)
# num_similar = st.sidebar.number_input("How many similar actives to show", min_value=1, max_value=20, value=5)

# st.sidebar.markdown("---")
# st.sidebar.markdown("**Notes:** Ensure RDKit is available in the environment.\nPlace the model file in the same folder as this script.")

# # ----------------------
# # Main layout
# # ----------------------
# st.title("🧪 Compound Bioactivity Predictor — Small GTPases")
# st.caption("Bioactivity prediction platform for drug discovery, powered by a calibrated Random Forest fingerprint model")

# col1, col2 = st.columns([1, 1])

# with col1:
#     st.subheader("Single-molecule prediction")
#     input_smiles = st.text_area("Enter SMILES (one molecule)", value="CCOC(=O)C1=CC2=CN=C(N=C2N1C)C1=CC=CC(=C1)C(F)(F)F", height=110)
#     st.markdown("**Example:** Paste a SMILES string then press Predict.")
#     predict_button = st.button("Predict")

# with col2:
#     st.subheader("Batch prediction")
#     uploaded_file = st.file_uploader("Upload CSV with a column named 'smiles' or 'canonical_smiles'", type=["csv"] )
#     st.markdown("CSV output will contain: smiles, canonical_smiles, probability_active, predicted_label, similarity_max, similarity_avg")
#     batch_run = st.button("Predict batch")

# st.markdown("---")

# # ----------------------
# # Single prediction flow
# # ----------------------
# if predict_button and input_smiles.strip():
#     with st.spinner("Processing molecule..."):
#         csmi = canonical_smiles(input_smiles.strip())
#         if not csmi:
#             st.error("Invalid SMILES — couldn't parse.")
#         else:
#             mol = mol_from_smiles(csmi)
#             fp_np = mol_to_fp_numpy(mol, radius=fp_radius, nbits=fp_nbits)
#             if fp_np is None:
#                 st.error("Failed to compute fingerprint.")
#             else:
#                 proba = None
#                 try:
#                     proba_all = model.predict_proba(np.array([fp_np]))
#                     classes = list(model.classes_)
#                     if 'active' in classes:
#                         idx_active = classes.index('active')
#                     else:
#                         # fallback: assume positive label encoded as 1
#                         idx_active = 1 if len(classes)>1 else 0
#                     proba = float(proba_all[0, idx_active])
#                 except Exception as e:
#                     st.error(f"Model prediction failed: {e}")

#                 if proba is not None:
#                     predicted_label = "active" if proba >= threshold else "inactive"

#                     # Similarity summary
#                     similarities = []
#                     for afp_rd in active_fps_rd:
#                         if afp_rd is None:
#                             continue
#                         try:
#                             sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol, fp_radius), afp_rd)
#                         except Exception:
#                             sim = 0.0
#                         similarities.append(sim)

#                     sim_arr = np.array(similarities) if similarities else np.array([0.0])
#                     sim_max = float(np.max(sim_arr))
#                     sim_avg = float(np.mean(sim_arr))

#                     st.metric("Predicted label", predicted_label.upper())
#                     st.metric("Probability (active)", f"{proba:.3f}")
#                     st.metric("Max similarity to train actives", f"{sim_max:.3f}")

#                     # Show molecule image and structure
#                     img_bytes = render_molecule_image(csmi, size=(420,240))
#                     if img_bytes:
#                         st.image(img_bytes, caption=f"Canonical SMILES: {csmi}")

#                     # Show similar actives
#                     if show_similar and len(active_smiles_list)>0:
#                         st.write(f"Top {num_similar} similar actives from training set")
#                         top_idx = np.argsort(-sim_arr)[:num_similar]
#                         rows = []
#                         for i, idx in enumerate(top_idx):
#                             if idx >= len(active_smiles_list):
#                                 continue
#                             s = active_smiles_list[idx]
#                             rows.append({"rank": i+1, "canonical_smiles": s, "similarity": float(sim_arr[idx])})
#                         sim_df = pd.DataFrame(rows)
#                         st.dataframe(sim_df)

# # ----------------------
# # Batch prediction flow
# # ----------------------
# if batch_run and uploaded_file is not None:
#     with st.spinner("Reading CSV and predicting — this may take a while for large files..."):
#         try:
#             df_in = pd.read_csv(uploaded_file)
#         except Exception as e:
#             st.error(f"Unable to read CSV: {e}")
#             df_in = None

#         if df_in is not None:
#             # find smiles column
#             smi_col = None
#             for col in ['canonical_smiles', 'smiles', 'SMILES', 'inchi']:
#                 if col in df_in.columns:
#                     smi_col = col
#                     break
#             if smi_col is None:
#                 st.error("CSV must contain a 'smiles' or 'canonical_smiles' column")
#             else:
#                 # prepare output
#                 out_rows = []
#                 for i, row in df_in.iterrows():
#                     s = row.get(smi_col, "")
#                     csmi = canonical_smiles(s) if pd.notna(s) else None
#                     if not csmi:
#                         out_rows.append({smi_col: s, 'canonical_smiles': None, 'probability_active': None, 'predicted_label': 'invalid_smiles', 'similarity_max': None, 'similarity_avg': None})
#                         continue
#                     mol = mol_from_smiles(csmi)
#                     fp_np = mol_to_fp_numpy(mol, radius=fp_radius, nbits=fp_nbits)
#                     if fp_np is None:
#                         out_rows.append({smi_col: s, 'canonical_smiles': csmi, 'probability_active': None, 'predicted_label': 'fp_failed', 'similarity_max': None, 'similarity_avg': None})
#                         continue
#                     try:
#                         proba_all = model.predict_proba(np.array([fp_np]))
#                         classes = list(model.classes_)
#                         if 'active' in classes:
#                             idx_active = classes.index('active')
#                         else:
#                             idx_active = 1 if len(classes)>1 else 0
#                         proba = float(proba_all[0, idx_active])
#                     except Exception as e:
#                         proba = None

#                     # similarity
#                     similarities = []
#                     for afp_rd in active_fps_rd:
#                         if afp_rd is None:
#                             continue
#                         try:
#                             sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol, fp_radius), afp_rd)
#                         except Exception:
#                             sim = 0.0
#                         similarities.append(sim)
#                     sim_arr = np.array(similarities) if similarities else np.array([0.0])
#                     sim_max = float(np.max(sim_arr))
#                     sim_avg = float(np.mean(sim_arr))

#                     predicted_label = 'active' if proba is not None and proba>=threshold else 'inactive' if proba is not None else 'error'

#                     out_rows.append({smi_col: s, 'canonical_smiles': csmi, 'probability_active': proba, 'predicted_label': predicted_label, 'similarity_max': sim_max, 'similarity_avg': sim_avg})

#                 out_df = pd.DataFrame(out_rows)
#                 st.success(f"Predicted {len(out_df)} molecules")
#                 st.dataframe(out_df)

#                 # download
#                 csv = out_df.to_csv(index=False)
#                 b64 = base64.b64encode(csv.encode()).decode()
#                 href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions.csv</a>'
#                 st.markdown(href, unsafe_allow_html=True)

# # ----------------------
# # Model explainability & diagnostics
# # ----------------------
# st.markdown("---")
# st.subheader("Model diagnostics & tips")
# col_a, col_b = st.columns(2)
# with col_a:
#     st.markdown("- Threshold is derived from PR-F1 during training but you can adjust it here.\n- 'Probability (active)' is the calibrated probability from the model.\n- Use similarity values to see how close a query molecule is to training actives.")
# with col_b:
#     st.markdown("- Invalid SMILES will be flagged.\n- Batch predictions can take time depending on file size and model complexity.")

# st.markdown("---")
# st.caption("Built for researchers and product teams ")  
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from rdkit import Chem
# from rdkit.Chem import AllChem, Draw
# from rdkit import DataStructs
# from io import BytesIO
# from PIL import Image
# import base64

st.set_page_config(page_title="Compound Bioactivity Predictor", page_icon="🧪", layout="wide")

# ----------------------
# Helpers
# ----------------------
@st.cache_resource
def load_model(path="rf_fp_model_pro.pkl"):
    try:
        data = joblib.load(path)
        return data
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None

@st.cache_data
def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

@st.cache_data
def canonical_smiles(smiles: str):
    m = mol_from_smiles(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m else None

@st.cache_data
def mol_to_fp_numpy(mol, radius=2, nbits=2048):
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

@st.cache_data
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def render_molecule_image(smiles, size=(350, 200)):
    m = mol_from_smiles(smiles)
    if not m:
        return None
    img = Draw.MolToImage(m, size=size)
    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

# ----------------------
# Load model
# ----------------------
with st.spinner("Loading model..."):
    model_data = load_model()

if not model_data:
    st.stop()

model = model_data.get("rf_fp_model_pro.pkl")
default_threshold = float(model_data.get("threshold", 0.5))   # Used internally, not adjustable in UI
fp_radius = int(model_data.get("fp_radius", 2))
fp_nbits = int(model_data.get("fp_nbits", 2048))
active_smiles_list = model_data.get("active_canonical_smiles", [])

# Precompute RDKit bitvectors for actives for fast similarity
active_mols = [mol_from_smiles(smi) for smi in active_smiles_list]
active_fps_rd = [AllChem.GetMorganFingerprint(m, fp_radius) if m else None for m in active_mols]

# ----------------------
# Sidebar: Settings (threshold control removed)
# ----------------------
st.sidebar.header("Settings")
show_similar = st.sidebar.checkbox("Show top similar actives", value=True)
num_similar = st.sidebar.number_input(
    "How many similar actives to show", min_value=1, max_value=20, value=5
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Notes:** Ensure RDKit is available in the environment.\n"
    "Place the model file in the same folder as this script."
)

# ----------------------
# Main layout
# ----------------------
st.title("🧪 Compound Bioactivity Predictor — Small GTPases")
st.caption("Bioactivity prediction platform for drug discovery, powered by a calibrated Random Forest fingerprint model")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single-molecule prediction")
    input_smiles = st.text_area(
        "Enter SMILES (one molecule)",
        value="CCOC(=O)C1=CC2=CN=C(N=C2N1C)C1=CC=CC(=C1)C(F)(F)F",
        height=110
    )
    st.markdown("**Example:** Paste a SMILES string then press Predict.")
    predict_button = st.button("Predict")

with col2:
    st.subheader("Batch prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV with a column named 'smiles' or 'canonical_smiles'", type=["csv"]
    )
    st.markdown(
        "CSV output will contain: smiles, canonical_smiles, probability_active, "
        "predicted_label, similarity_max, similarity_avg"
    )
    batch_run = st.button("Predict batch")

st.markdown("---")

# ----------------------
# Single prediction flow
# ----------------------
if predict_button and input_smiles.strip():
    with st.spinner("Processing molecule..."):
        csmi = canonical_smiles(input_smiles.strip())
        if not csmi:
            st.error("Invalid SMILES — couldn't parse.")
        else:
            mol = mol_from_smiles(csmi)
            fp_np = mol_to_fp_numpy(mol, radius=fp_radius, nbits=fp_nbits)
            if fp_np is None:
                st.error("Failed to compute fingerprint.")
            else:
                proba = None
                try:
                    proba_all = model.predict_proba(np.array([fp_np]))
                    classes = list(model.classes_)
                    if 'active' in classes:
                        idx_active = classes.index('active')
                    else:
                        idx_active = 1 if len(classes) > 1 else 0
                    proba = float(proba_all[0, idx_active])
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")

                if proba is not None:
                    predicted_label = "active" if proba >= default_threshold else "inactive"

                    # Similarity summary
                    similarities = []
                    for afp_rd in active_fps_rd:
                        if afp_rd is None:
                            continue
                        try:
                            sim = DataStructs.TanimotoSimilarity(
                                AllChem.GetMorganFingerprint(mol, fp_radius), afp_rd
                            )
                        except Exception:
                            sim = 0.0
                        similarities.append(sim)

                    sim_arr = np.array(similarities) if similarities else np.array([0.0])
                    sim_max = float(np.max(sim_arr))
                    sim_avg = float(np.mean(sim_arr))

                    st.metric("Predicted label", predicted_label.upper())
                    st.metric("Probability (active)", f"{proba:.3f}")
                    st.metric("Max similarity to train actives", f"{sim_max:.3f}")

                    img_bytes = render_molecule_image(csmi, size=(420, 240))
                    if img_bytes:
                        st.image(img_bytes, caption=f"Canonical SMILES: {csmi}")

                    if show_similar and len(active_smiles_list) > 0:
                        st.write(f"Top {num_similar} similar actives from training set")
                        top_idx = np.argsort(-sim_arr)[:num_similar]
                        rows = []
                        for i, idx in enumerate(top_idx):
                            if idx >= len(active_smiles_list):
                                continue
                            s = active_smiles_list[idx]
                            rows.append({
                                "rank": i + 1,
                                "canonical_smiles": s,
                                "similarity": float(sim_arr[idx])
                            })
                        sim_df = pd.DataFrame(rows)
                        st.dataframe(sim_df)

# ----------------------
# Batch prediction flow
# ----------------------
if batch_run and uploaded_file is not None:
    with st.spinner("Reading CSV and predicting — this may take a while for large files..."):
        try:
            df_in = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")
            df_in = None

        if df_in is not None:
            smi_col = None
            for col in ['canonical_smiles', 'smiles', 'SMILES', 'inchi']:
                if col in df_in.columns:
                    smi_col = col
                    break
            if smi_col is None:
                st.error("CSV must contain a 'smiles' or 'canonical_smiles' column")
            else:
                out_rows = []
                for i, row in df_in.iterrows():
                    s = row.get(smi_col, "")
                    csmi = canonical_smiles(s) if pd.notna(s) else None
                    if not csmi:
                        out_rows.append({
                            smi_col: s, 'canonical_smiles': None,
                            'probability_active': None,
                            'predicted_label': 'invalid_smiles',
                            'similarity_max': None, 'similarity_avg': None
                        })
                        continue
                    mol = mol_from_smiles(csmi)
                    fp_np = mol_to_fp_numpy(mol, radius=fp_radius, nbits=fp_nbits)
                    if fp_np is None:
                        out_rows.append({
                            smi_col: s, 'canonical_smiles': csmi,
                            'probability_active': None,
                            'predicted_label': 'fp_failed',
                            'similarity_max': None, 'similarity_avg': None
                        })
                        continue
                    try:
                        proba_all = model.predict_proba(np.array([fp_np]))
                        classes = list(model.classes_)
                        if 'active' in classes:
                            idx_active = classes.index('active')
                        else:
                            idx_active = 1 if len(classes) > 1 else 0
                        proba = float(proba_all[0, idx_active])
                    except Exception:
                        proba = None

                    similarities = []
                    for afp_rd in active_fps_rd:
                        if afp_rd is None:
                            continue
                        try:
                            sim = DataStructs.TanimotoSimilarity(
                                AllChem.GetMorganFingerprint(mol, fp_radius), afp_rd
                            )
                        except Exception:
                            sim = 0.0
                        similarities.append(sim)
                    sim_arr = np.array(similarities) if similarities else np.array([0.0])
                    sim_max = float(np.max(sim_arr))
                    sim_avg = float(np.mean(sim_arr))

                    predicted_label = (
                        'active' if proba is not None and proba >= default_threshold
                        else 'inactive' if proba is not None else 'error'
                    )

                    out_rows.append({
                        smi_col: s,
                        'canonical_smiles': csmi,
                        'probability_active': proba,
                        'predicted_label': predicted_label,
                        'similarity_max': sim_max,
                        'similarity_avg': sim_avg
                    })

                out_df = pd.DataFrame(out_rows)
                st.success(f"Predicted {len(out_df)} molecules")
                st.dataframe(out_df)

                csv = out_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions.csv</a>'
                st.markdown(href, unsafe_allow_html=True)

# ----------------------
# Model explainability & diagnostics
# ----------------------
st.markdown("---")
st.subheader("Model diagnostics & tips")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        "- Threshold is fixed from training (stored in the model file).\n"
        "- 'Probability (active)' is the calibrated probability from the model.\n"
        "- Use similarity values to see how close a query molecule is to training actives."
    )
with col_b:
    st.markdown(
        "- Invalid SMILES will be flagged.\n"
        "- Batch predictions can take time depending on file size and model complexity."
    )

st.markdown("---")
st.caption("Built for researchers and product teams")
