from fastai.vision.all import *
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix, classification_report

# --- Configuration ---
path_folder = Path("/home/pglabns04/Downloads/deep-cervical-cancer-master/sipakmed_formatted")
model_path = path_folder/'hybrid_densenet121_vit_model'/'export.pkl'
test_dataset_path = path_folder/'test'

print("Loading learner and preparing inference...")

# 1. Load the Learner (v2 uses the same load_learner function)
learn = load_learner(model_path)

# 2. Get all test files
# FastAI v2 provides get_image_files to recursively grab images
test_files = get_image_files(test_dataset_path)

# 3. Create a DataLoader for the test set
# This is much faster than a manual 'for' loop for predictions
test_dl = learn.dls.test_dl(test_files, with_labels=True)

# 4. Get Predictions
# 'with_decoded' returns the specific class index (argmax) automatically
preds, targets, decoded = learn.get_preds(dl=test_dl, with_decoded=True)

# 5. Extract results
# targets are the actual labels, decoded are the predicted indices
predictions = decoded.numpy()
labels = targets.numpy()

# 6. Calculate Metrics
# Calculate mistakes manually to match your original script logic
mistakes = (predictions != labels).sum()
total_size_samples = len(labels)
accuracy = (1 - (mistakes / total_size_samples)) * 100.0

# --- Output Results ---
print(f"The accuracy of the testing dataset is {accuracy:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nClassification Report:")
# In v2, learn.dls.vocab stores the class names
print(classification_report(labels, predictions, target_names=learn.dls.vocab))