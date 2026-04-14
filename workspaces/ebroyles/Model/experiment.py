"""



Show Train loss vs epoch and Train accuracy vs epoch
0,1,2,other = (3,4,5,6,7,8)
even the dataset (4000, 2000, 5500, 1200) boost all to 3000 by cut to 64 x 64, norm onto 0 to 1, use time mixup (ie grab a chunk and move it to the other side), and also use square dropout with gaussian random noise between 0 and 1, also use class weighting so being wrong on the other class is really bad, and being 

do a study that find the distribution of dB readings inside the spectograms
show train confusion matrix
show test confusion matrix
show time to complete
show the newly created data and distributions


"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

from collections import defaultdict
# load all the old data
# convert it to an image, normalize it



class Experiment:
    DATA_ROOT = "../../fiot_highway2-main/data/"
    TRAIN_TXT = "../../fiot_highway2-main/train.txt"
    TEST_TXT  = "../../fiot_highway2-main/test.txt"
    DB_MIN = -136
    DB_MAX = 0

    def __init__(self, min_label=0, max_label=8):
        self.min_label = min_label
        self.max_label = max_label

    def _path_label_struct_extract(self, struct):
        paths, labels = [], []
        for entry in struct:
            label = int(entry[1])
            if self.min_label <= label <= self.max_label:
                path = entry[0].split('/')[1].split('.')[0]
                paths.append(path)
                labels.append(label)
        return paths, labels

    def get_train_test_paths_labels(self, target_labels=list(range(0, 9))):
        train_struct = np.loadtxt(self.TRAIN_TXT, dtype=str).tolist()
        test_struct  = np.loadtxt(self.TEST_TXT,  dtype=str).tolist()
    
        train_paths, train_labels = self._path_label_struct_extract(train_struct)
        test_paths,  test_labels  = self._path_label_struct_extract(test_struct)
    
        if target_labels is not None:
            target_labels = set(target_labels) 
    
            def filter_by_labels(paths, labels):
                filtered_paths = [p for p, l in zip(paths, labels) if l in target_labels]
                filtered_labels = [l for l in labels if l in target_labels]
                return filtered_paths, filtered_labels
    
            train_paths, train_labels = filter_by_labels(train_paths, train_labels)
            test_paths, test_labels = filter_by_labels(test_paths, test_labels)
    
        return train_paths, train_labels, test_paths, test_labels

    def _plot_locations_by_label(self, locs, labels, title, chunk_size=3):
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10")  # 10 distinct colors
        
        # Split labels into chunks of chunk_size
        for i in range(0, len(unique_labels), chunk_size):
            chunk = unique_labels[i:i+chunk_size]
            
            plt.figure(figsize=(16, 12))
            
            for label in chunk:
                idx = labels == label
                plt.scatter(locs[idx, 1], locs[idx, 0],  # x=time, y=freq
                            color=cmap(label % 10),
                            label=f"Class {label}",
                            alpha=0.7,
                            edgecolors='w',
                            s=50)
            
            plt.xlabel("Time bin")
            plt.ylabel("Frequency bin")
            plt.title(f"{title} (Classes {chunk[0]} to {chunk[-1]})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def dB_distribution_study(self):
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels()

        paths  = train_paths + test_paths
        labels = train_labels + test_labels
        total_paths = len(paths)

        mins, maxs = [], []
        min_locs, max_locs = [], []

        for i, path in enumerate(paths):
            arr = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
            min_val = arr.min()
            max_val = arr.max()
            min_idx = np.unravel_index(arr.argmin(), arr.shape)
            max_idx = np.unravel_index(arr.argmax(), arr.shape)
            mins.append(min_val)
            maxs.append(max_val)
            min_locs.append(min_idx)
            max_locs.append(max_idx)
            print(f"{i}/{total_paths}", end='\r')

        mins = np.array(mins)
        maxs = np.array(maxs)
        min_locs = np.array(min_locs)
        max_locs = np.array(max_locs)
        labels = np.array(labels)

        print("Global min dB:", mins.min())
        print("Global max dB:", maxs.max())

        plt.scatter(labels, mins, label="Min dB", alpha=0.6)
        plt.scatter(labels, maxs, label="Max dB", alpha=0.6)
        plt.legend()
        plt.grid(True)
        plt.show()

        self._plot_locations_by_label(min_locs, labels, "Min dB Locations by Label")
        self._plot_locations_by_label(max_locs, labels, "Max dB Locations by Label")

    def std_distribution(self, label):
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels(target_labels=[label])
        paths = train_paths + test_paths
    
        col_mean = None
        col_median = None
        col_std = None
    
        count = 0  # number of samples processed
    
        # For median and std, collect all to compute at the end (if memory permits)
        all_col_medians = []
        all_col_stds = []
    
        for path in paths:
            arr = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
    
            specto_col_mean = np.mean(arr, axis=0)
            specto_col_median = np.median(arr, axis=0)
            specto_col_std = np.std(arr, axis=0)
    
            if col_mean is None:
                col_mean = np.zeros_like(specto_col_mean)
            col_mean = col_mean * (count / (count + 1)) + specto_col_mean / (count + 1)
    
            all_col_medians.append(specto_col_median)
            all_col_stds.append(specto_col_std)
    
            count += 1
            print(f"Processing {count}/{len(paths)}", end='\r')
    
        # Compute median of medians and mean of stds
        col_median = np.median(np.vstack(all_col_medians), axis=0)
        col_std = np.mean(np.vstack(all_col_stds), axis=0)
    
        # Plot results
        import matplotlib.pyplot as plt
        x = np.arange(col_mean.size)
    
        plt.figure(figsize=(12, 6))
        plt.plot(x, col_mean, label="Mean (running avg)")
        plt.plot(x, col_median, label="Median (of medians)")
        plt.plot(x, col_std, label="Std (mean of stds)")
        plt.xlabel("Column Index")
        plt.ylabel("Value")
        plt.title(f"Column-wise Mean, Median, and Std for label {label}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_std_dev(self):
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels()

        paths  = train_paths + test_paths
        labels = train_labels + test_labels
        total_paths = len(paths)

        std_dev = []
        
        for i, path in enumerate(paths):
            arr = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
            std_dev.append(np.std(arr))
            print(f"{i}/{total_paths}", end='\r')

        std_dev = np.array(std_dev)
        labels = np.array(labels)

        plt.scatter(labels, std_dev, label="Standard Deviations", alpha=0.6)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_mean(self):
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels()

        paths  = train_paths + test_paths
        labels = train_labels + test_labels
        total_paths = len(paths)

        mean = []
        
        for i, path in enumerate(paths):
            arr = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
            mean.append(np.mean(arr))
            print(f"{i}/{total_paths}", end='\r')

        mean = np.array(mean)
        labels = np.array(labels)

        plt.scatter(labels, mean, label="Mean", alpha=0.6)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_median(self):
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels()

        paths  = train_paths + test_paths
        labels = train_labels + test_labels
        total_paths = len(paths)

        median = []
        
        for i, path in enumerate(paths):
            arr = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
            median.append(np.median(arr))
            print(f"{i}/{total_paths}", end='\r')

        median = np.array(median)
        labels = np.array(labels)

        plt.scatter(labels, median, label="Median", alpha=0.6)
        plt.legend()
        plt.grid(True)
        plt.show()

    def give_me_a_specto(self, label):
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels()
        paths = train_paths + test_paths
        labels = train_labels + test_labels
    
        # Filter paths by given label
        filtered_paths = [p for p, l in zip(paths, labels) if l == label]
        if not filtered_paths:
            print(f"No spectrogram found with label {label}")
            return
    
        # Randomly select one
        chosen_path = random.choice(filtered_paths)
        arr = np.load(os.path.join(self.DATA_ROOT, chosen_path + ".npy"))

        print(f"Label: {label}")
        print(f"Selected spectrogram: {chosen_path}")
        print(f"Min: {np.min(arr)}")
        print(f"Max: {np.max(arr)}")
        print(f"Median: {np.median(arr)}")
        print(f"Mean: {np.mean(arr)}")
        print(f"Standard Deviation: {np.std(arr)}")

    def dB_histograms(self, target_labels):
        print("start histogram")
        train_paths, train_labels, test_paths, test_labels = self.get_train_test_paths_labels()
        paths = train_paths #+ test_paths
        labels = train_labels #+ test_labels
        filtered_paths = [p for p, l in zip(paths, labels) if l in target_labels]
        filtered_labels = [l for l in labels if l in target_labels]

        paths = filtered_paths
        labels = filtered_labels
        total_paths = len(paths)


        
        
        class_values = defaultdict(list)
        i = 0
        for path, label in zip(paths, labels):
            arr = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
            # flatten the array to 1D
            class_values[label].extend(arr.flatten())
            print(f"{i}/{total_paths}", end='\r')
            i += 1
    
        plt.figure(figsize=(15, 10))
        num_classes = len(set(labels))
        bins = 30
        for cls in sorted(class_values.keys()):
            plt.hist(class_values[cls], bins=bins, alpha=0.5, label=f"Class {cls}", density=True)
    
        plt.xlabel("dB Value")
        plt.ylabel("Density")
        plt.title("dB Value Distribution per Class")
        plt.legend()
        plt.grid(True)
        plt.show()


    def 
            
            
            












# class HighwayDataset(Dataset):
#     def __init__(self, paths, labels, data_root, resize=(64,64)):
#         self.paths = paths
#         self.labels = labels
#         self.data_root = data_root
#         self.resize = resize

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         fname = self.paths[idx] + ".npy"
#         arr = np.load(os.path.join(self.data_root, fname))   
#         arr = np.clip(arr, MIN_DB, MAX_DB) # Clip to a fixed dB range 
#         arr = (arr - MIN_DB) / (MAX_DB - MIN_DB) # Scale to [0,1] VERIFY THIS
#         img = torch.from_numpy(arr.astype(np.float32))    # H x W
#         img = img.unsqueeze(0)                            # 1 Input Channel (amplitude) || 1 x H x W
#         img = TF.resize(img, self.resize)                 # RESIZE still 1 x h x w
#         label = int(self.labels[idx]) - MIN_LABEL # **** Label shift 
#         return img, label


