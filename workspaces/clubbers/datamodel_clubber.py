# --- Setup & Install radiomana ---
# Run this once per notebook session.

# Option 1 (recommended): Install directly from GitHub
pip install git+https://github.com/the-aerospace-corporation/radiomana.git
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Imports ---
import os
import torch
import radiomana
import matplotlib.pyplot as plt

# --- Set the dataset environment variable ---
folder_path = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
os.environ["DSET_FIOT_HIGHWAY2"] = folder_path

print("✅ radiomana installed and dataset path set!")
print("radiomana version:", radiomana.__version__)
print("CUDA available:", torch.cuda.is_available())
# data path to train.txt file
train_file = os.path.join(folder_path, "train.txt")

# read contents of train.txt
with open(train_file, "r") as f:
    lines = f.readlines()

# i will display the first few lines of the file to inspect its structure
for line in lines[:10]:  # Show first 10 lines for example
    print(line.strip())
#now that I know how the data is structured...
#to let myself see what I am getting myself into, I am going to randomly plot 5 sample files
#and observe their spectrogram. I want to understand how the frequencies work in the data.

import random
import matplotlib.pyplot as plt

n_samples = 5  # ima pick 5 random samples

# this is sampling random files from dataset
random_files = random.sample(npy_files, n_samples)

# these are specific plot grid settings
fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 6), constrained_layout=True)

for ax, fname in zip(axes, random_files):
    # this should the data for each sample
    data = np.load(os.path.join(data_folder, fname))
    
    # plotting the spectrogram
    im = ax.imshow(data, origin="lower", aspect="auto")
    ax.set_title(fname, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

plt.show()

# a quick note, I understand I haven't done any preprocessing yet. I just want to get a rough idea of what I am
# dealing with
import os
import matplotlib.pyplot as plt

# my file path defined
train_file = os.path.join(folder_path, "train.txt")

# need an empty dictionary for all my values later
label_counts = {}

# insert values into the previous label_counts
with open(train_file, "r") as f:
    for line in f:
        path, label = line.strip().split()
        label = int(label)  # Convert label to integer
        label_counts[label] = label_counts.get(label, 0) + 1  # Increment count for this label

# print the classes + counts per
for label, count in sorted(label_counts.items()):
    print(f"{label}: {count}")

# plot distribution in bar chart using dictionary
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.title("Class Distribution in Train Split")
plt.show()
import numpy as np
import os
import matplotlib.pyplot as plt

folder_path = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
train_file = os.path.join(folder_path, "train.txt")

# this will be used to read train.txt file + store data like class labels
class_files = {i: [] for i in range(9)}  # i set range to 9 classes (since 0 to 8)

with open(train_file, "r") as f:
    for line in f:
        file_path, label = line.strip().split()
        class_files[int(label)].append(file_path) 

# a function to load + plot one sample of a specific class (I want the first that appears)
def plot_class(class_id):
    sample_data = [] # list will be used later to put the file in
    file_path = class_files[class_id][0]  # only first sample (for now. I added this feature if I wanted more in the future)
    file_path = os.path.join(folder_path, file_path)
    
    try: # quick try-except just in case there is a corrupt file :D
        print(f"Looking at File: {file_path}") # i want to see which file i am looking at
        data = np.load(file_path)
        sample_data.append(data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    
    # plotting time
    fig, axes = plt.subplots(1, 1, figsize=(4, 4)) # i set it to 4, 4 to see better
    axes.imshow(sample_data[0], aspect='auto', cmap="viridis")
    axes.set_title(f"Class {class_id} Sample")
    axes.set_xlabel("Time")
    axes.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Plot data for each class
for class_id in range(9):  # 9 classes (0 to 8)
    plot_class(class_id)

