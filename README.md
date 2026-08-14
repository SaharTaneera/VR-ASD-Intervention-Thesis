Markdown
# VR-Based Motion Analysis & Intervention System for ASD

Official repository for my Master's thesis research in Computer Engineering at Istanbul Medipol University. This project presents an interactive, game-based virtual reality motion-analysis and intervention system designed to assist children with Autism Spectrum Disorder (ASD), validated through a clinical pilot trial.

## 🧠 System Overview & Architecture
The system evaluates user movement accuracy in a VR environment by comparing skeletal tracking data against expert references. It combines two complementary approaches:
1. **Siamese Transformer Network (`siamese_traine.py`):** Deep learning architecture with self-attention and data augmentations to distinguish correct movement execution (Heian Yondan / HY-1) from contrastive incorrect or different movements (Bassai Dai / BD-1).
2. **Classical DTW & Geometric Scoring (`kata_scoring.py`):** Preprocessing pipeline handling missing value interpolation, pelvis centering, orientation normalization, scaling, Savitzky–Golay smoothing, and Dynamic Time Warping (DTW) distance mapping.
3. **FastAPI Backend (`api.py`):** Real-time evaluation endpoint providing automated progress scores, limb-specific breakdowns, segment windows, and motion-energy penalty calculations.

## 🥽 VR Integration & Data Pipeline
* **Meta Quest (Oculus) Integration:** Live motion readings and tracking data were captured during gameplay via an Oculus/Meta Quest headset.
* **Kinematic Estimation:** Because standard headset and controller setups provide limited points, the pipeline utilized kinematic estimation to compute and map full-body joint markers (corresponding to Qualisys optical mocap standards).
* **Unity Client:** While the core AI backend and scoring models are open-sourced here, the proprietary Unity game client code remains private. The FastAPI backend serves as the bridge, accepting real-time TSV session logs from the game client and returning instant evaluation metrics.

## 📂 Repository Structure

├── api.py                      # FastAPI server for real-time session evaluation

├── siamese_traine.py           # Siamese Transformer network and data pipelines

├── kata_scoring.py             # Classical DTW scoring and geometric normalization

├── train_model.py              # Script to fit expert baseline models

├── evaluate_siamese_model.py   # Cross-validation and evaluation metrics script

└── requirements.txt            # Python dependencies

## ⚙️ Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/your-username/VR-ASD-Intervention-Thesis.git](https://github.com/your-username/VR-ASD-Intervention-Thesis.git)
cd VR-ASD-Intervention-Thesis
Install the required dependencies:

Bash
pip install -r requirements.txt

## 🚀 Usage
1. Training / Fitting Expert References
To fit the reference model using expert motion trials:

Bash
python train_model.py

2. Running the Evaluation Suite
To execute cross-validation and evaluate model performance against test splits:

Bash
python evaluate_siamese_model.py --device cuda

3. Launching the Backend API
To start the FastAPI server for real-time evaluation:

Bash
uvicorn api:app --host 0.0.0.0 --port 8000

📊 Dataset & Organization
The core model training and evaluation utilize optical motion capture data (Qualisys TSV format recording 3D coordinates of 25 body joints).

Public Dataset Source: You can download the underlying karate motion capture dataset (featuring expert katas like Heian Yondan and Bassai Dai) from the EyesWeb Karate Dataset Repository. (http://www.infomus.org/karate/eyesweb_dataset_karate_eng.php)

Local Directory Structure: Organize your downloaded TSV files locally for execution as follows:

Plaintext
C:\Users\ROG\dataset\
├── HY-1\    # Target / Reference movement (Heian Yondan)

└── BD-1\    # Contrastive / Negative class (Bassai Dai)

## 📜 Citation
If you find this research useful for your work, please cite the repository or related thesis publications, as well as the original dataset source:

'''bibtex
@article{niewiadomski2018analysis,
  author   = {Niewiadomski, Radoslaw and Kolykhalova, Ksenia and Piana, Stefania and Alborno, Paolo and Volpe, Gualtiero and Camurri, Antonio},
  title    = {Analysis of Movement Quality in Full-Body Physical Activities},
  journal  = {ACM Transactions on Interactive Intelligent Systems},
  volume   = {9},
  number   = {1},
  articleno = {3},
  numpages = {20},
  year     = {2018},
  publisher = {Association for Computing Machinery},
  address  = {New York, NY, USA},
  doi      = {10.1145/3132369}
}
'''
