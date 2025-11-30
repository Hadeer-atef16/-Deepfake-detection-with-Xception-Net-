# -Deepfake-detection-with-Xception-Net-
Deepfake image classification using XceptionNet on CelebDF-V2 dataset.
This repository provides a complete pipeline for Deepfake image detection using the XceptionNet model trained on the CelebDF-V2 dataset.
The project includes preprocessing, model training, evaluation, and inference scripts — fully optimized and Colab-friendly.

🚀 Features

✔ Face extraction using MTCNN

✔ Binary classification: Real vs Fake

✔ XceptionNet architecture (pretrained on ImageNet)

✔ Full training & validation loops

✔ Metrics: Accuracy, AUC, Loss curves

✔ Ready-to-use inference script

✔ Compatible with Google Colab

📁 Project Structure
deepfake-detection/
│── data/                # dataset path (CelebDF-V2)
│── preprocessing/        # face cropping & MTCNN scripts
│── models/               # trained model files (.h5 / .pth)
│── notebooks/            # Google Colab notebooks
│── scripts/              # training & inference python files
│── requirements.txt
│── README.md
│── .gitignore

📦 Requirements

Install dependencies:

pip install -r requirements.txt

Main Libraries:

TensorFlow / Keras

facenet-pytorch (MTCNN)

OpenCV

NumPy, Pandas, Matplotlib

Scikit-learn

🎯 Dataset: CelebDF-V2

CelebDF-V2 is a high-quality Deepfake dataset with realistic and challenging fake samples.

Dataset link:
https://github.com/yuezunli/celeb-deepfakeforensics

After downloading, extract frames or images and place them inside:

data/

🏋️‍♂️ Training

To train the XceptionNet model:

python scripts/train_xception.py --epochs 50 --batch_size 32


Or run the training notebook:

notebooks/Deepfake_Xception_Training.ipynb


Training pipeline includes:

Loading real & fake images

Preprocessing & face alignment

Building XceptionNet

Validation loop

Saving best model based on validation AUC

🔍 Inference (Predict Single Image)

Run the inference script:

python scripts/predict.py --image path_to_image.jpg


Output example:

Real (12%)  or  Fake (83%)

📊 Results

The XceptionNet model achieved high performance on the CelebDF-V2 test set.

Confusion Matrix
                 Predicted Real    Predicted Fake
True Real           3828               10
True Fake           11                 3829

Metrics

Correct Predictions: 7657 / 7678

Test Accuracy: 0.9973 (≈ 99.73%)

Model Behavior:

Very low False Positives → 11

Very low False Negatives → 10

Balanced performance across both classes

📝 Summary

Model: XceptionNet

Dataset: CelebDF-V2

Achieved Accuracy: ~99.73%

Strong generalization across multiple identities

Excellent separation between real & fake images

🔮 Future Work

Test EfficientNet-B4

Add GradCAM for explainability

Add SHAP + PCA visualizations

Improve face alignment preprocessing

Add video-level detection pipeline (frame scoring + aggregation)
