#trial
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd # For better table representation
import seaborn as sns # For heatmaps

# Force debugging sync
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'epochs': 1, # Number of epochs for local training
    'batch_size': 32,
    'initial_lr': 0.001,
}

# --- Dataset Paths ---
# Using the same dataset paths as before
train_dataset_paths = {
    "D1": "/kaggle/input/chestxraydataset/chest_xray/train",
    "D2": "/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train",
    "D3": "/kaggle/input/pediatric-pneumonia-chest-xray/Pediatric Chest X-ray Pneumonia/train",
    "D4": "/kaggle/input/chest-xray-covid19-pneumonia/Data/train",
    "D5": "/kaggle/input/pneumonia-tuberculosis-normal/Train",
}

test_dataset_paths = {
    "D1": "/kaggle/input/chestxraydataset/chest_xray/test",
    "D2": "/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test",
    "D3": "/kaggle/input/pediatric-pneumonia-chest-xray/Pediatric Chest X-ray Pneumonia/test",
    "D4": "/kaggle/input/chest-xray-covid19-pneumonia/Data/test",
    "D5": "/kaggle/input/pneumonia-tuberculosis-normal/Test",
}

# --- Data Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Function to Load Filtered Dataset ---
def load_filtered_dataset(path):
    allowed_classes = ['NORMAL', 'PNEUMONIA']
    dataset = datasets.ImageFolder(path, transform=transform)

    class_idx_map = {name: i for i, name in enumerate(allowed_classes)}
    
    filtered_samples = []
    for img_path, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_name in allowed_classes:
            new_label = class_idx_map[class_name]
            filtered_samples.append((img_path, new_label))

    dataset.samples = filtered_samples
    dataset.targets = [label for _, label in filtered_samples]
    dataset.classes = allowed_classes
    dataset.class_to_idx = class_idx_map

    return DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

# --- Simple CNN Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2), # 2 classes: NORMAL, PNEUMONIA
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# --- Training Function for Centralized Learning ---
def train_model_centralized(model, dataloader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['initial_lr'])
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Training for {epochs} epochs...")
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
    return model

# --- Evaluation Function (similar to before) ---
def evaluate_model_centralized(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

# --- Centralized Learning Main Loop ---
print("--- Starting Centralized Learning Experiments ---")

# This will store accuracy results for the heatmap
cross_dataset_accuracies = pd.DataFrame(index=train_dataset_paths.keys(), columns=test_dataset_paths.keys(), dtype=float)

for train_client_id, train_path in train_dataset_paths.items():
    print(f"\n--- Training a NEW model on dataset: {train_client_id} ---")
    
    # Initialize a new model for each training run
    model = SimpleCNN().to(device)
    
    # Load training data for the current client
    train_loader = load_filtered_dataset(train_path)
    
    # Train the model on this specific dataset
    trained_model = train_model_centralized(model, train_loader, CONFIG['epochs'])
    
    # --- Evaluate the trained model on ALL test datasets ---
    print(f"  Evaluating model trained on {train_client_id} across all test datasets:")
    for test_client_id, test_path in test_dataset_paths.items():
        test_loader = load_filtered_dataset(test_path)
        true_labels, predictions = evaluate_model_centralized(trained_model, test_loader)
        
        # Calculate accuracy for the heatmap
        accuracy = np.mean(np.array(true_labels) == np.array(predictions))
        cross_dataset_accuracies.loc[train_client_id, test_client_id] = accuracy
        print(f"    Tested on {test_client_id}: Accuracy = {accuracy:.4f}")


print("\n--- Centralized Learning Experiments Completed ---")

# --------------- Visualizations for Centralized Learning ---------------

# --- 1. Cross-Dataset Classification Accuracy Heatmap ---
print("\n--- Generating Cross-Dataset Accuracy Heatmap ---")
plt.figure(figsize=(12, 10))
sns.heatmap(cross_dataset_accuracies, annot=True, fmt=".4f", cmap="viridis",
            cbar_kws={'label': 'Accuracy'}, linewidths=.5, linecolor='black')
plt.title('Cross-Dataset Classification Accuracy Heatmap\n(Rows: Trained On, Columns: Tested On)', fontsize=20)
plt.xlabel('Tested On Dataset', fontsize=12)
plt.ylabel('Trained On Dataset', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 2. Summary Table of Max/Min Accuracies for each trained model ---
print("\n--- Summary Table of Performance for Each Trained Model ---")
trained_model_summary = []
for train_client_id in cross_dataset_accuracies.index:
    row_data = cross_dataset_accuracies.loc[train_client_id]
    trained_model_summary.append({
        'Trained On': train_client_id,
        'Avg Acc (Across Tests)': row_data.mean(),
        'Min Acc': row_data.min(),
        'Max Acc': row_data.max(),
        'Acc on Self': row_data.loc[train_client_id] # Accuracy when tested on its own dataset
    })
df_summary = pd.DataFrame(trained_model_summary).set_index('Trained On')
print(df_summary.to_string(float_format="%.4f"))

# --- 3. Bar Chart: Accuracy when tested on own dataset ---
print("\n--- Bar Chart: Accuracy on Self-Test ---")
plt.figure(figsize=(8, 6))
plt.bar(df_summary.index, df_summary['Acc on Self'], color='lightcoral')
plt.xlabel('Dataset Trained On', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy of Model when Tested on its Own Training Dataset', fontsize=20)
plt.ylim(0, 1)
for i, acc in enumerate(df_summary['Acc on Self']):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 4. Bar Chart: Average Accuracy Across All Test Datasets (for each trained model) ---
print("\n--- Bar Chart: Average Generalization Accuracy ---")
plt.figure(figsize=(8, 6))
plt.bar(df_summary.index, df_summary['Avg Acc (Across Tests)'], color='mediumseagreen')
plt.xlabel('Dataset Trained On', fontsize=12)
plt.ylabel('Average Accuracy', fontsize=12)
plt.title('Average Accuracy of Model (Trained on X) Across All Test Datasets', fontsize=20)
plt.ylim(0, 1)
for i, acc in enumerate(df_summary['Avg Acc (Across Tests)']):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n--- All centralized learning results and visualizations generated. ---")
