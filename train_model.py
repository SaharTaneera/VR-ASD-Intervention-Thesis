# train_model.py
from glob import glob
from kata_scoring import load_qualisys_tsv, learn_expert_model, MARKERS
import pickle, os

expert_dir = r"C:\Users\ROG\dataset\HY-1"   
expert_paths = sorted(glob(expert_dir + r"\*.tsv"))

expert_dfs = [load_qualisys_tsv(p, MARKERS) for p in expert_paths]
model = learn_expert_model(expert_dfs)

with open(os.path.join(expert_dir, "hy_model.pkl"), "wb") as f:
    pickle.dump(model, f)
print("Saved model.")
