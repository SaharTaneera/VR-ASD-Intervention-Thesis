import argparse
import os
import sys
import zipfile
import shutil
import types
import glob
import json
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


def extract_zip(zip_path, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def patch_tensorboard_if_missing():
    try:
        import tensorboard  # noqa
    except Exception:
        tb_mod = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def add_figure(self, *args, **kwargs):
                pass

            def close(self):
                pass

        tb_mod.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tb_mod


def find_dataset_dirs(root):
    hy_dirs = glob.glob(os.path.join(root, "**", "HY-1"), recursive=True)
    bd_dirs = glob.glob(os.path.join(root, "**", "BD-1"), recursive=True)

    if not hy_dirs or not bd_dirs:
        raise FileNotFoundError(
            "Could not find HY-1 and BD-1 folders. Check the extracted dataset structure."
        )

    hy_dir = hy_dirs[0]
    bd_dir = bd_dirs[0]

    hy_files = sorted(glob.glob(os.path.join(hy_dir, "*.tsv")))
    bd_files = sorted(glob.glob(os.path.join(bd_dir, "*.tsv")))

    return hy_files, bd_files


def evaluate_fold(st, model_path, hy_val, bd_val, fold, device, batch_size=8):
    model = st.SiameseNet(J=len(st.MARKERS), C=6, d_model=128).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    val_ds = st.StratifiedPairsDataset(hy_val, bd_val, cache=True)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []
    sims = []

    with torch.no_grad():
        for X1, m1, X2, m2, y in loader:
            X1 = X1.to(device)
            m1 = m1.to(device)
            X2 = X2.to(device)
            m2 = m2.to(device)

            _, _, sim = model(X1, m1, X2, m2)

            pred = (sim.view(-1) > 50).cpu().numpy().astype(int)

            y_true.extend(y.numpy().astype(int).tolist())
            y_pred.extend(pred.tolist())
            sims.extend(sim.cpu().numpy().reshape(-1).tolist())

    labels = [1, 0]  
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    score_df = pd.DataFrame(
        {
            "fold": fold,
            "true_label": y_true,
            "pred_label": y_pred,
            "similarity_score": sims,
            "pair_type": ["HY-HY" if y == 1 else "HY-BD" for y in y_true],
        }
    )

    summary = {
        "fold": fold,
        "n_pairs": len(y_true),
        "accuracy": acc,
        "cm_HYHY_predHYHY": int(cm[0, 0]),
        "cm_HYHY_predHYBD": int(cm[0, 1]),
        "cm_HYBD_predHYHY": int(cm[1, 0]),
        "cm_HYBD_predHYBD": int(cm[1, 1]),
        "HYHY_precision": precision[0],
        "HYHY_recall": recall[0],
        "HYHY_f1": f1[0],
        "HYHY_support": int(support[0]),
        "HYBD_precision": precision[1],
        "HYBD_recall": recall[1],
        "HYBD_f1": f1[1],
        "HYBD_support": int(support[1]),
        "HYHY_mean_score": float(score_df.loc[score_df["true_label"] == 1, "similarity_score"].mean()),
        "HYBD_mean_score": float(score_df.loc[score_df["true_label"] == 0, "similarity_score"].mean()),
    }

    return summary, score_df, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_zip", default="dataset.zip")
    parser.add_argument("--runs_zip", default="runs.zip")
    parser.add_argument("--code_dir", default=".")
    parser.add_argument("--out_dir", default="eval_outputs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        args.device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    work_dir = os.path.join(args.out_dir, "_extracted")
    dataset_dir = os.path.join(work_dir, "dataset")
    runs_dir = os.path.join(work_dir, "runs")

    print("Extracting dataset...")
    extract_zip(args.dataset_zip, dataset_dir)

    print("Extracting runs...")
    extract_zip(args.runs_zip, runs_dir)

    patch_tensorboard_if_missing()
    sys.path.insert(0, args.code_dir)

    import siamese_traine as st

    hy_files, bd_files = find_dataset_dirs(dataset_dir)

    print(f"Found {len(hy_files)} Heian Yondan files")
    print(f"Found {len(bd_files)} Bassai Dai files")

    pd.DataFrame(
        {
            "kata": ["Heian Yondan", "Bassai Dai"],
            "n_files": [len(hy_files), len(bd_files)],
            "role": ["target/reference", "contrastive/dissimilar"],
        }
    ).to_csv(os.path.join(args.out_dir, "dataset_summary.csv"), index=False)

    all_files_labels = [(f, 1) for f in hy_files] + [(f, 0) for f in bd_files]
    labels = [lab for _, lab in all_files_labels]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_summaries = []
    all_scores = []
    all_y_true = []
    all_y_pred = []

    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        hy_val = []
        bd_val = []

        for idx in val_idx:
            f, lab = all_files_labels[idx]
            if lab == 1:
                hy_val.append(f)
            else:
                bd_val.append(f)

        model_path = os.path.join(
            runs_dir,
            "runs",
            "siamese_cv",
            f"fold{fold}",
            "best_model.pth",
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        print(f"\nEvaluating fold {fold}...")
        summary, score_df, cm = evaluate_fold(
            st,
            model_path,
            hy_val,
            bd_val,
            fold,
            args.device,
            args.batch_size,
        )

        all_summaries.append(summary)
        all_scores.append(score_df)

        all_y_true.extend(score_df["true_label"].tolist())
        all_y_pred.extend(score_df["pred_label"].tolist())

        np.savetxt(
            os.path.join(args.out_dir, f"confusion_matrix_fold{fold}.csv"),
            cm,
            delimiter=",",
            fmt="%d",
        )

        print(f"Fold {fold} accuracy: {summary['accuracy']:.3f}")
        print(cm)

    summary_df = pd.DataFrame(all_summaries)
    scores_df = pd.concat(all_scores, ignore_index=True)

    summary_df.to_csv(os.path.join(args.out_dir, "fold_summary.csv"), index=False)
    scores_df.to_csv(os.path.join(args.out_dir, "all_pair_scores.csv"), index=False)

    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=[1, 0])
    overall_acc = accuracy_score(all_y_true, all_y_pred)

    np.savetxt(
        os.path.join(args.out_dir, "confusion_matrix_overall.csv"),
        overall_cm,
        delimiter=",",
        fmt="%d",
    )

    report = classification_report(
        all_y_true,
        all_y_pred,
        labels=[1, 0],
        target_names=["HY-HY similar", "HY-BD dissimilar"],
        digits=4,
        zero_division=0,
    )

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    overall = {
        "overall_accuracy": overall_acc,
        "mean_fold_accuracy": float(summary_df["accuracy"].mean()),
        "std_fold_accuracy": float(summary_df["accuracy"].std()),
        "overall_confusion_matrix_labels": ["HY-HY", "HY-BD"],
        "overall_confusion_matrix": overall_cm.tolist(),
        "HYHY_mean_similarity": float(scores_df.loc[scores_df["true_label"] == 1, "similarity_score"].mean()),
        "HYBD_mean_similarity": float(scores_df.loc[scores_df["true_label"] == 0, "similarity_score"].mean()),
    }

    with open(os.path.join(args.out_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\n================ OVERALL RESULTS ================")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Mean fold accuracy: {overall['mean_fold_accuracy']:.4f} ± {overall['std_fold_accuracy']:.4f}")
    print("Overall confusion matrix, labels = [HY-HY, HY-BD]:")
    print(overall_cm)
    print("\nClassification report:")
    print(report)
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
