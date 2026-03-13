# README: Offensive and Nasty Hate Speech Detection from Roman Urdu Social Media Platforms

## Title
An Ensemble Approach for Offensive and Nasty Hate Speech Detection from Roman Urdu Social Media Platforms

## Description
This repository contains the source code, datasets, and experimental configurations for our research on detecting offensive and hate speech written in Roman Urdu using machine learning and deep learning techniques. The repository supports reproducibility and transparency for academic review and public use.

## Dataset Information
We used three benchmark datasets:
- **RUHSOLD** – 20,000 tweets labeled as hate or neutral (GitHub 2019)
- **HS-RU-20** – Binary labeled dataset with various hate categories (Kaggle 2020)
- **VVD-21** – Tweets classified as violent or non-violent (Kaggle 2021)

Each dataset is preprocessed and split into training and testing files for use in classification tasks.

## Code Information
- `preprocessing.py` – Cleans and tokenizes raw Roman Urdu tweets.
- `feature_extraction.py` – Extracts TF-IDF, N-gram, POS tags, sentiment scores, and word embeddings.
- `train_models.py` – Trains six classifiers: Naive Bayes, Random Forest, SVM, Logistic Regression, LSTM, and MLP.
- `evaluation.py` – Runs 5-fold cross-validation, cross-dataset testing, and ablation experiments.
- `visualization.py` – Generates bar charts, confusion matrices, and performance plots.

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/HiraAkhtar-Butt/Offensive-and-Nasty-hate-Speech-Detection-from-Roman-Urdu-Social-Media-Platforms.git
   cd Offensive-and-Nasty-hate-Speech-Detection-from-Roman-Urdu-Social-Media-Platforms
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Preprocess datasets:
   ```bash
   python preprocessing.py
   ```
4. Train models:
   ```bash
   python train_models.py --model lstm --dataset ruhsold
   ```
5. Evaluate performance:
   ```bash
   python evaluation.py
   ```

## Requirements
- Python 3.9+
- TensorFlow 2.12+
- scikit-learn 1.3+
- pandas, numpy, nltk, matplotlib, seaborn

## Methodology
- Text preprocessing (cleaning, tokenization)
- Feature extraction (TF-IDF, word embeddings, POS tagging, sentiment)
- Model training (NB, RF, SVM, LR, LSTM, MLP)
- Evaluation methods: 5-fold cross-validation, cross-dataset validation, ablation study

## Citations
If you use this code, please cite our paper:
> Hira Akhtar Butt et al. (2025). An Ensemble Approach for Offensive and Nasty Hate Speech Detection from Roman Urdu Social Media Platforms.

## License
This project is released under the MIT License.

## Contribution Guidelines
Pull requests and forks are welcome for performance improvements, additional datasets, or model integrations. Please open an issue first to discuss any major changes you wish to propose.
