import os
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, ConcatDataset

LABELS = {0,1,2,3,4,5,6,7,8}
SHAPE = (512, 243)
COL_AXIS, ROW_AXIS = (0, 1)

"""
Data Saved As:
specto0: np rowwise array [X, y]
...

"""

class MySavedDataset(Dataset):
    SAVE_ROOT = "my_data/"
        
    def __init__(self, path):
        self.filepath = os.path.join(self.SAVE_ROOT, path)
        self.data = np.load(self.filepath, mmap_mode="r") #lazy_load_data
        
    def __len__(self):
        return self.data.shape[0] #rows

    def __getitem__(self, idx):
        row = torch.from_numpy(self.data[idx].copy())
        X = row[:-1].float()
        y = row[-1].long()   
        return X, y

    def get_num_features(self):
        return self.data.shape[1]-1 #cols why is this 1345 and not 1343

class MyDataset(Dataset):
    DATA_ROOT = "../../fiot_highway2-main/data/"
    TRAIN_TXT = "../../fiot_highway2-main/train.txt"
    TEST_TXT  = "../../fiot_highway2-main/test.txt"
    
    def __init__(self, filepath_txt = None, get_features_func = None, resize_shape = SHAPE, labels = LABELS):
        
        # filepath_txt: TRAIN_TXT or TEST_TXT
        self.labels = labels
        self.filepath_txt = filepath_txt
        self.get_features_func = get_features_func
        self.resize_shape = resize_shape
        self.specto_paths = []
        self.specto_labels = []
        self.set_specto_paths_labels()
        self.num_specto = len(self.specto_paths)
        X,_ = self[0]
        self.num_features = len(X)

    def __len__(self):
        return self.num_specto

    def __getitem__(self, idx):
        if self.get_features_func is None: return None
        specto_path = self.specto_paths[idx]
        y = self.specto_labels[idx]
        specto = self.get_specto(specto_path)
        X = self.get_features_func(specto, self.resize_shape)
        return X, y
            
    def set_specto_paths_labels(self):
        # txt_filepath: either TRAIN_TXT or TEST_TXT or smothing matching their format
        # creates: paths == ["000000", ...], labels == [0, ...]
        struct = np.loadtxt(self.filepath_txt, dtype=str).tolist()
        for entry in struct:
            label = int(entry[1])
            if label in self.labels:
                path = entry[0].split('/')[1].split('.')[0]
                self.specto_paths.append(path)
                self.specto_labels.append(label)

    def get_specto(self, path):
        # path: "000000"
        return np.load(os.path.join(self.DATA_ROOT, path + ".npy"))

    def save_all_items(self, path, norm_type = 0):
        #norm type can be None, 0,1,...
        all_items = []
        for i in range(len(self)):
            X, y = self[i]
            row = np.hstack((X, y))
            all_items.append(row)
            print(f"{i}/{len(self)-1}", end='\r')
            print()
        all_items = np.vstack(all_items)  # shape (num_samples, num_features + 1)
        match norm_type:
            case 0: pass
            case 1: all_items = self.normz(all_items)
            case 2: all_items = self.norm0to1(all_items)
        save_path = os.path.join(MySavedDataset.SAVE_ROOT, path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, all_items)
        return all_items
        
    def normz(self, items):
        """
        Z-score normalize features column-wise.
        items: np.ndarray, shape (N, F + 1)
        """
        features = items[:, :-1].astype(np.float32)
        labels = items[:, -1:] #no change
        mean = features.mean(axis=COL_AXIS, keepdims=True)
        std = features.std(axis=COL_AXIS, keepdims=True)
        std[std == 0] = 1.0  # avoid div by zero
        features_norm = (features - mean) / std
        items_norm = np.hstack((features_norm, labels))
        return items_norm
        
    def norm0to1(self, items):
        """
        Min-max normalize features column-wise to [0, 1].
        items: np.ndarray, shape (N, F + 1)
        """
        items = items.astype(np.float32, copy=False)
        features = items[:, :-1]
        labels = items[:, -1:]
        min_val = features.min(axis=COL_AXIS, keepdims=True)
        max_val = features.max(axis=COL_AXIS, keepdims=True)
        denom = max_val - min_val
        denom[denom == 0] = 1.0  # avoid divide-by-zero
        features_norm = (features - min_val) / denom
        return np.hstack((features_norm, labels))
        

class SpectoFeatures:
    
    @staticmethod
    def get_features_f1(specto, resize_shape):
        specto = SpectoFeatures.resize_specto(specto, resize_shape)
        specto = SpectoFeatures.flip_specto(specto)
        X = []  
        X.append(SpectoFeatures.mean_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.mean_specto(specto, ROW_AXIS))
        X.append(SpectoFeatures.standard_deviation_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.standard_deviation_specto(specto, ROW_AXIS))
        X.append(SpectoFeatures.median_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.median_specto(specto, ROW_AXIS))
        X.append(SpectoFeatures.min_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.min_specto(specto, ROW_AXIS))
        X.append(SpectoFeatures.max_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.max_specto(specto, ROW_AXIS))
        X.append(SpectoFeatures.min_location_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.min_location_specto(specto, ROW_AXIS))
        X.append(SpectoFeatures.max_location_specto(specto, COL_AXIS))
        X.append(SpectoFeatures.max_location_specto(specto, ROW_AXIS))
        X = np.hstack(X)
        return X

    @staticmethod
    def resize_specto(specto, shape):
        # shape (height, width)
        return resize(specto, shape, anti_aliasing=True, preserve_range=True)

    @staticmethod
    def flip_specto(specto):
        # multiply specto by -1 to change range of dB from 0 to -136 to 0 to 136
        return -specto
        
    @staticmethod
    def mean_specto(specto, axis=None):
        return np.atleast_1d(np.mean(specto, axis))

    @staticmethod
    def standard_deviation_specto(specto, axis=None):
        return np.atleast_1d(np.std(specto, axis))

    @staticmethod
    def median_specto(specto, axis=None):
        return np.atleast_1d(np.median(specto, axis))

    @staticmethod
    def min_specto(specto, axis=None):
        return np.atleast_1d(np.min(specto, axis))

    @staticmethod
    def max_specto(specto, axis=None):
        return np.atleast_1d(np.max(specto, axis))

    @staticmethod
    def min_location_specto(specto, axis=None):
        return np.atleast_1d(np.argmin(specto, axis))

    @staticmethod
    def max_location_specto(specto, axis=None):
        return np.atleast_1d(np.argmax(specto, axis))




        




        
        


        



"""




AGH
"""
# class MyDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray):
#         """
#         X: shape (number of spectograms, number of spectogram features)
#         y: (labels) shape (number of spectograms,)
#         """
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

#     def norm(self):
#         # norm onto 0 to 1 for each feature (ie along the columns)
#         # Min and max for each feature (column)
#         min_vals = self.X.min(dim=0).values
#         max_vals = self.X.max(dim=0).values
        
#         # Avoid division by zero by replacing zeros with ones
#         range_vals = max_vals - min_vals
#         range_vals[range_vals == 0] = 1.0
        
#         # Normalize each feature to [0, 1]
#         self.X = (self.X - min_vals) / range_vals
        
# class DataManager:
#     """
#     Assumes:
#     * Path to data file: Data_ROOT + 000000.npy
#     * contents of TRAIN_TXT and TEST_TXT are 
#         path            label
#         data/000000.npy 0
#         ...             ...
#     """
#     DATA_ROOT = "../../fiot_highway2-main/data/"
#     TRAIN_TXT = "../../fiot_highway2-main/train.txt"
#     TEST_TXT  = "../../fiot_highway2-main/test.txt"

#     def extract_paths_labels(self, txt_filepath, min_label = 0, max_label = 8):
#         # txt_filepath: either TRAIN_TXT or TEST_TXT or smothing matching their format
#         # returns: paths == ["000000", ...], labels == [0, ...]
#         struct = np.loadtxt(txt_filepath, dtype=str).tolist()
#         paths, labels = [], []
#         for entry in struct:
#             label = int(entry[1])
#             if min_label <= label <= max_label:
#                 path = entry[0].split('/')[1].split('.')[0]
#                 paths.append(path)
#                 labels.append(label)
#         return paths, labels

#     def load_np_data(self, path):
#         # path: "000000"
#         # returns: np [[], ...], spectorgram 243 x 512 (c,r)
#         specto = np.load(os.path.join(self.DATA_ROOT, path + ".npy"))
#         return specto

#     def flip_specto(self, specto):
#         # multiply specto by -1 to change range of dB from 0 to -136 to 0 to 136
#         return -specto
        
#     def mean_specto(self, specto, axis = None):
#         # specto: np [[], ...]
#         # axis: None == full mean, 0 == colwise, 1 == rowwise
#         # returns: np row vector of means
#         return np.hstack(np.mean(specto, axis))

#     def standard_deviation_specto(self, specto, axis = None):
#         return np.hstack(np.std(specto, axis))

#     def median_specto(self, specto, axis = None):
#         return np.hstack(np.median(specto, axis))

#     def min_specto(self, specto, axis = None):
#         return np.hstack(np.min(specto, axis))

#     def max_specto(self, specto, axis = None):
#         return np.hstack(np.max(specto, axis))

#     def min_location_specto(self, specto, axis = None):
#         return np.hstack(np.argmin(specto, axis))

#     def max_location_specto(self, specto, axis = None):
#         return np.hstack(np.argmax(specto, axis))

#     def get_X_v1(self, path, resize_shape, get_num_features = False):
#         COL_AXIS, ROW_AXIS = (0, 1)
#         specto = self.load_np_data(path)
#         specto = resize(specto, resize_shape, anti_aliasing=True, preserve_range=True)
#         specto = self.flip_specto(specto)
#         X = []  
#         X.append(self.mean_specto(specto, COL_AXIS))
#         X.append(self.mean_specto(specto, ROW_AXIS))
#         X.append(self.standard_deviation_specto(specto, COL_AXIS))
#         X.append(self.standard_deviation_specto(specto, ROW_AXIS))
#         X.append(self.median_specto(specto, COL_AXIS))
#         X.append(self.median_specto(specto, ROW_AXIS))
#         X.append(self.min_specto(specto, COL_AXIS))
#         X.append(self.min_specto(specto, ROW_AXIS))
#         X.append(self.max_specto(specto, COL_AXIS))
#         X.append(self.max_specto(specto, ROW_AXIS))
#         X.append(self.min_location_specto(specto, COL_AXIS))
#         X.append(self.min_location_specto(specto, ROW_AXIS))
#         X.append(self.max_location_specto(specto, COL_AXIS))
#         X.append(self.max_location_specto(specto, ROW_AXIS))
#         X = np.hstack(X)
#         if get_num_features: 
#             num_features = len(X)
#             return X, num_features
#         return X

#     def get_dataloader_v1(self, resize_shape = (128, 64), get_num_features = False):
#         # resize_shape = (h,w)
#         train_paths, train_labels = self.extract_paths_labels(self.TRAIN_TXT)
#         test_paths, test_labels   = self.extract_paths_labels(self.TEST_TXT)
#         len_train, len_test = len(train_paths), len(test_paths) #number of spectorgrams in train and test
#         print("len_train: ", len_train)
#         print("len_test: ", len_test)
#         _, num_features = self.get_X_v1(train_paths[0], resize_shape, get_num_features = True)
#         print("num_features: ", num_features)
#         train_y, test_y = np.array(train_labels), np.array(test_labels)
#         train_X, test_X = np.zeros((len_train, num_features)), np.zeros((len_test, num_features))
#         for i, path in enumerate(train_paths + test_paths):
#             X = self.get_X_v1(path, resize_shape)
#             if i < len_train: train_X[i, :] = X
#             else: test_X[i - len_train, :] = X
#             print(f"{i}/{len_train+len_test}", end='\r')
#             ## very slow

#         train_ds = MyDataset(train_X, train_y)
#         test_ds = MyDataset(test_X, test_y)
#         train_ds.norm()
#         test_ds.norm()
#         train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=63)
#         test_loader = DataLoader(test_ds, batch_size=64)
#         if get_num_features: return train_loader, test_loader, num_features
#         return train_loader, test_loader

        
            
        
        


        

    
        

    

    
    