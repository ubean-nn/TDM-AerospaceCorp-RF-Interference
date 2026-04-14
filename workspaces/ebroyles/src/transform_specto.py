import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from constants import SAVE_ROOT, DATA_ROOT, TRAIN_TXT, TEST_TXT, LABELS


"""

transform_train = TS(is_test=False, pre_process_funcs = [], augment_funcs = [], extract_features_funcs = [], batchsize = 64) -> df = None, batchedpaths and labels created
transform_train.populate_df() -> runs pre_process, augment, extract features -> populates df
transform_test = ...
transform_test.populate_df()
TS.normalize(transform_train, transform_test) or TS.normalize(transform_train)
transform_train.save_df(path="my_data/data0") #save to data0
transform_test.save_df(path="my_data/data0") #save to 



TS.save(transform_train, transform_test) or TS.normalize(transform_train)
TS.save_description(transform_train, transform_test) or ...

"""

class TransformSpecto:

    def __init__(self, is_test=False, include_labels=LABELS, times=1, pre_process_funcs=[], augment_funcs=[], features_extract_funcs=[], batchsize=64):
        self.is_test = is_test
        self.include_labels = include_labels
        self.times = times
        self.pre_process_funcs = pre_process_funcs
        self.augment_funcs = augment_funcs
        self.features_extract_funcs = features_extract_funcs
        self.batchsize = batchsize
        self.df = None
        self.batches_paths = []
        self.batches_labels = []
        self._normalized_with = TransformSpecto.no_norm
        self._set_batches()

    def _set_batches(self):
        paths, labels = self._get_paths_labels()
        def batched(arr, size=64): return [arr[i:i+size] for i in range(0, len(arr), size)]
        s = self.batchsize
        self.batches_paths, self.batches_labels = batched(paths, s), batched(labels, s)
            
    def _get_paths_labels(self):
        source = TEST_TXT if self.is_test else TRAIN_TXT
        paths, labels = [], []
        for row in np.loadtxt(source, dtype=str).tolist():
            label = int(row[1])
            if label in self.include_labels:
                path = row[0].split('/')[1].split('.')[0]
                for time in range(self.times): 
                    paths.append(path)
                    labels.append(label)
        return paths, labels

    def populate_df(self):
        rows = []; batch = 1; num_batches = len(self.batches_paths)
        for batch_paths, batch_labels in zip(self.batches_paths, self.batches_labels):
            specto_batch = []
            for path in batch_paths: specto_batch.append(self._do_pre_process(TransformSpecto._get_spectogram(path)))
            self._do_augment(specto_batch)
            for specto, label in zip(specto_batch, batch_labels): rows.append(self._do_features_extract(specto) + [label])
            print(f"{batch}/{num_batches}", end="\r"); batch += 1
        print()
        df = pd.DataFrame(rows)
        self.df = df
        
    def save_df(self, foldername):
        filename = "test.csv" if self.is_test else "train.csv"
        savepath = os.path.join(SAVE_ROOT, foldername)
        os.makedirs(savepath, exist_ok=True)
        print(f"saving to: {savepath}")
        if self.df is None: return
        self.df.columns = [f"f{i}" for i in range(self.df.shape[1] - 1)] + ["label"] #header
        self.df.to_csv(os.path.join(savepath, filename), index=False)

    def save_description(self, foldername):
        filename = "description.txt"
        savepath = os.path.join(SAVE_ROOT, foldername)
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, filename)

        num_features = self.df.shape[1] - 1  
        total = len(self.df)
    
        def fn_names(fns):
            if not fns: return "None"
            names = []
            for f in fns:
                # Handle functools.partial objects vs regular functions
                if hasattr(f, 'func'): names.append(f.func.__name__)
                else: names.append(f.__name__)
            return ", ".join(names)
    
        with open(savepath, "w") as f:
            f.write("Dataset Summary\n")
            f.write("----------------\n")
            f.write(f"Number of features : {num_features}\n")
            f.write(f"Total samples for train.csv: {total}\n")
            f.write(f"Include Labels     : {self.include_labels}\n")
            f.write(f"Times              : {self.times}\n")
            f.write("Spectogram -> Features\n")
            f.write("----------------------\n")
            f.write(f"Pre-processing   : {fn_names(self.pre_process_funcs)}\n")
            f.write(f"Augment          : {fn_names(self.augment_funcs)}\n")
            f.write(f"Feature extract  : {fn_names(self.features_extract_funcs)}\n")
            f.write(f"Normalization    : {fn_names([self._normalized_with])}\n")

    def _do_pre_process(self, specto):
        new_specto = specto
        for f in self.pre_process_funcs: new_specto = f(new_specto)
        return new_specto

    def _do_augment(self, specto_batch):
        new_specto_batch = specto_batch
        for f in self.augment_funcs: new_specto_batch = f(new_specto_batch)
        return new_specto_batch
        
    def _do_features_extract(self, specto):
        features = []
        for f in self.features_extract_funcs: features.extend(f(specto))
        return features

    @staticmethod
    def _get_spectogram(path):
        return np.load(os.path.join(DATA_ROOT, path + ".npy"))

    @staticmethod
    def _display_augmented_specto(specto1, specto2, title1="Original", title2="Augmented"):
        assert specto1.ndim == 2, "specto1 must be a 2D array (F, T)"
        assert specto2.ndim == 2, "specto2 must be a 2D array (F, T)"
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
        im0 = axes[0].imshow(specto1, aspect="auto", origin="lower")
        axes[0].set_title(title1)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")
    
        im1 = axes[1].imshow(specto2, aspect="auto", origin="lower")
        axes[1].set_title(title2)
        axes[1].set_xlabel("Time")
    
        # Shared colorbar
        plt.colorbar(im0, ax=axes, location="right", shrink=0.8)
        plt.show()
        
    """
    Pre Process
    """
    @staticmethod
    def resize_specto(specto, shape):
        # shape (height, width)
        return resize(specto, shape, anti_aliasing=True, preserve_range=True)

    @staticmethod
    def flip_specto(specto):
        # multiply specto by -1 to change range of dB from 0 to -136 to 0 to 136
        return -specto

    """
    Augment 
    """
    @staticmethod
    def time_mixup_specto(specto_batch, num_spectos_to_mix=3, display_sample=False):
        """
        Time-domain mixup with EVEN proportions.
        Each of K spectrograms contributes ~time_bins / K columns.
        Assumes all spectrograms have the same shape and label.
        """
        num_spectos = len(specto_batch)
        if num_spectos < num_spectos_to_mix: return specto_batch
        time_bins = specto_batch[0].shape[1] # specto shape: (freq_bins, time_bins)
        base_width = time_bins // num_spectos_to_mix  # even share
        remainder = time_bins % num_spectos_to_mix    # leftover columns
        augmented_batch = []
        for _ in range(num_spectos):
            time_cursor = 0
            chunks = []
            for i, idx in enumerate(np.random.randint(num_spectos, size=num_spectos_to_mix)):
                # last spectrogram takes the remainder
                width = base_width + (remainder if i == num_spectos_to_mix - 1 else 0)
                specto = specto_batch[idx]
                chunk = specto[:, time_cursor:time_cursor + width]
                chunks.append(chunk)
                time_cursor += width
            augmented = np.concatenate(chunks, axis=1)
            augmented_batch.append(augmented)
        if display_sample:
            idx = np.random.randint(num_spectos)
            TransformSpecto._display_augmented_specto(specto_batch[idx], augmented_batch[idx], title2="time_mixup_specto")
        return augmented_batch

    @staticmethod
    def rect_dropout_specto(specto_batch, max_freq_frac=0.1, max_time_frac=0.2, display_sample=False):
        ## UNTESTED
        #draw a rectangle of noise on each specto
        B, F, T = specto_batch.shape
        augmented = specto_batch.copy()
    
        noise_min = specto_batch.min()
        noise_max = specto_batch.max()
    
        for b in range(B):
            fh = np.random.randint(1, int(F * max_freq_frac) + 1) #height
            tw = np.random.randint(1, int(T * max_time_frac) + 1) #width
            f0 = np.random.randint(0, F - fh) #freq_pos
            t0 = np.random.randint(0, T - tw) #time_pos
            noise = np.random.uniform(noise_min, noise_max, size=(fh, tw))
            augmented[b, f0:f0+fh, t0:t0+tw] = noise

        if display_sample: 
            idx = np.random.randint(0, B)  # random index in [0, B-1]
            TransformSpecto._display_augmented_specto(specto_batch[idx], augmented[idx], title2="rect_dropout_specto")
        return augmented
        
    """
    Features Extract
    """
    @staticmethod
    def mean(specto, axis=None):
        return np.atleast_1d(np.mean(specto, axis))

    @staticmethod
    def std(specto, axis=None):
        return np.atleast_1d(np.std(specto, axis))

    @staticmethod
    def median(specto, axis=None):
        return np.atleast_1d(np.median(specto, axis))

    @staticmethod
    def min(specto, axis=None):
        return np.atleast_1d(np.min(specto, axis))

    @staticmethod
    def max(specto, axis=None):
        return np.atleast_1d(np.max(specto, axis))

    @staticmethod
    def minloc(specto, axis=None):
        return np.atleast_1d(np.argmin(specto, axis))

    @staticmethod
    def maxloc(specto, axis=None):
        return np.atleast_1d(np.argmax(specto, axis))

    @staticmethod
    def iqr(specto, axis=None):
        q75 = np.percentile(specto, 75, axis=axis)
        q25 = np.percentile(specto, 25, axis=axis)
        return np.atleast_1d(q75 - q25)

    @staticmethod
    def percentile(specto, p=75, axis=None):
        return np.atleast_1d(np.percentile(specto, p, axis=axis))

    @staticmethod
    def flat_specto(specto):
        return specto.flatten()

    """
    Normalize
    """
    @staticmethod
    def normalize(func, transform_train=None, transform_test=None):

        train_df = transform_train.df if transform_train is not None else None
        test_df  = transform_test.df  if transform_test  is not None else None
        train_df, test_df = func(train_df, test_df)
        if transform_train is not None: transform_train.df = train_df; transform_train._normalized_with = func
        if transform_test  is not None: transform_test.df  = test_df; 

    @staticmethod
    def no_norm(train_df, test_df=None):
        return train_df, test_df

    @staticmethod
    def normz(train_df, test_df=None):
        """
        Z-score normalize using train statistics.
        Calculates mean and std for each feature column (all except the last).
        """
        if train_df is None:
            raise ValueError("train_df must not be None for normalization")
        features_idx = train_df.columns[:-1]
        train_mean = train_df[features_idx].mean()
        train_std = train_df[features_idx].std()
        train_std[train_std == 0] = 1.0
        new_train = train_df.copy()
        new_train[features_idx] = (train_df[features_idx] - train_mean) / train_std

        new_test = None
        if test_df is not None:
            new_test = test_df.copy()
            new_test[features_idx] = (test_df[features_idx] - train_mean) / train_std
        return new_train, new_test

    @staticmethod
    def norm0to1(train_df, test_df=None):
        """
        Min-max normalize using train statistics.
        Scales features to the range [0, 1].
        """
        if train_df is None:
            raise ValueError("train_df must not be None for normalization")
        features_idx = train_df.columns[:-1]
        train_min = train_df[features_idx].min()
        train_max = train_df[features_idx].max()
        denom = train_max - train_min
        denom[denom == 0] = 1.0 # Avoid division by zero
        new_train = train_df.copy()
        new_train[features_idx] = (train_df[features_idx] - train_min) / denom

        new_test = None
        if test_df is not None:
            new_test = test_df.copy()
            new_test[features_idx] = (test_df[features_idx] - train_min) / denom
        return new_train, new_test

















"""
my_data/standard0 -> train.csv, test.csv, description.txt
my_data/augment0 -> train.csv, description.txt

from functools import partial
labels = {0,1,2,3,4,5,6,7,8}
pre_process = [partial(TransformSpecto.flip), partial(TransformSpecto.resize, (128,64))]
feature_extract = [partial(TransformSpecto.mean, axis=0), partial(TransformSpecto.mean, axis=1), ...]
normalize = [partial(TransformSpecto.norm_zscore)]
ts = TransformSpecto(variant = TransformSpecto.STANDARD, pre_process, feature_extract, normalize)
ts.populate()
ts.save() -> saves the data, normalizes the data, writes the desription


labels = {0,1,8}
pre_process = [partial(TransformSpecto.flip), partial(TransformSpecto.resize, (128,64))]
feature_extract = [partial(TransformSpecto.mean, axis=0), partial(TransformSpecto.mean, axis=1), ...]
normalize = [partial(TransformSpecto.norm_zscore)]
ts = TransformSpecto(variant = TransformSpecto.AUGMENT, pre_process, feature_extract, normalize)
ts.populate(labels, times = 2)
ts.save() -> saves the data, normalizes the data, writes the desription

"""

# class TransformSpecto:
#     VARIANT_STANDARD = "standard"
#     VARIANT_AUGMENT = "augment"
    
#     def __init__(self, variant=None, pre_process = [], features_extract = [], normalize = []):
#         self.variant = variant
#         self.pre_process = pre_process         # functions to preprocess spectograms
#         self.features_extract = features_extract # functions to convert spectograms into features
#         self.normalize = normalize             # functions to normalize ALL features (at the same time) NOTE: norm train across features, then use same train norm parmas to norm test
#         self.train_specto_paths = []
#         self.train_specto_labels = []
#         self.test_specto_paths = []
#         self.test_specto_labels = []

#     def populate(self, filter_labels=LABELS, times=1):
#         self.train_specto_paths, self.train_specto_labels = TransformSpecto._populate(TRAIN_TXT, filter_labels, times) 
#         if self.variant == TransformSpecto.VARIANT_STANDARD: 
#             self.test_specto_paths, self.test_specto_labels = TransformSpecto._populate(TEST_TXT, filter_labels, times)

#     def save(self, path=None):
#         savepath = self._get_savepath(path)
#         print(f"saving to: {savepath}")
#         os.makedirs(savepath, exist_ok=True)
#         train_df = TransformSpecto._get_dataframe(self.train_specto_paths, self.train_specto_labels, self.pre_process, self.features_extract) 
#         if self.variant == TransformSpecto.VARIANT_STANDARD: 
#             test_df = TransformSpecto._get_dataframe(self.test_specto_paths, self.test_specto_labels, self.pre_process, self.features_extract)
#         else: test_df = None
#         train_df, test_df = TransformSpecto.do_normalize(train_df, test_df, self.normalize)
#         print("normalized df")
#         TransformSpecto._save_csv(savepath, "train.csv", train_df)
#         print("saved train")
#         TransformSpecto._save_csv(savepath, "test.csv", test_df)
#         print("saved test")
#         self._save_description(savepath, train_df, test_df)
#         print("saved desccription")

    

#     @staticmethod
#     def do_pre_process(spectogram, funcs = []):
#         # specto passed by reference all pre_process funcs directly modify it by reference.
#         new_specto = spectogram
#         for f in funcs: new_specto = f(new_specto)
#         return new_specto 

#     @staticmethod
#     def do_augment(spectogram_batch, funcs=[]):
        
        

#     @staticmethod
#     def do_features_extract(spectogram, funcs = []):
#         # spectogram is read only; features are appended to by each function
#         features = []
#         for f in funcs: features.append(f(spectogram))
#         return features
        
#     @staticmethod
#     def do_normalize(train_df, test_df = None, funcs = []):
#         new_train_df, new_tes_df = train_df, test_df
#         for f in funcs: new_train_df, new_tes_df = f(train_df, test_df)
#         return new_train_df, new_tes_df

#     @staticmethod
#     def _populate(filepath_txt, filter_labels=LABELS, times=1):
#         paths, labels = [], []
#         for row in np.loadtxt(filepath_txt, dtype=str).tolist():
#             label = int(row[1])
#             if label in filter_labels:
#                 path = row[0].split('/')[1].split('.')[0]
#                 for time in range(times): 
#                     paths.append(path)
#                     labels.append(label)
#         return paths, labels

#     @staticmethod
#     def _get_spectogram(path):
#         return np.load(os.path.join(DATA_ROOT, path + ".npy"))
    
#     def _get_savepath(self, path=None):
#         if path is not None: return path
#         if self.variant not in {TransformSpecto.VARIANT_STANDARD, TransformSpecto.VARIANT_AUGMENT}: raise ValueError(f"Unknown variant: {variant}")
#         os.makedirs(SAVE_ROOT, exist_ok=True)
#         count = 0
#         for name in os.listdir(SAVE_ROOT):
#             if name.startswith(self.variant):
#                 count += 1
#         return os.path.join(SAVE_ROOT, f"{self.variant}{count + 1}")

#     @staticmethod
#     def _get_dataframe(specto_paths, specto_labels, pre_process, features_extract):
#         #change this so it does stuff in batches, then 
#         rows = []
#         for i, (path, label) in enumerate(zip(specto_paths, specto_labels)):
#             specto = TransformSpecto._get_spectogram(path)
#             specto = TransformSpecto.do_pre_process(specto, pre_process)
            
#             X_list = TransformSpecto.do_features_extract(specto, features_extract)
#             flat_features = []
#             for item in X_list:
#                 if isinstance(item, np.ndarray):
#                     flat_features.extend(item.flatten())
#                 else:
#                     flat_features.append(item)
#             rows.append(flat_features + [label])
#             print(f"{i}/{len(specto_paths)-1}", end="\r")
#         print()
#         df = pd.DataFrame(rows)
#         return df
        
#     @staticmethod
#     def _save_csv(savepath, name, df = None):
#         os.makedirs(savepath, exist_ok=True)
#         if df is None: return
#         df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["label"] #header
#         df.to_csv(os.path.join(savepath, name), index=False)

#     def _save_description(self, savepath, train_df, test_df):
#         os.makedirs(savepath, exist_ok=True)
#         desc_path = os.path.join(savepath, "description.txt")
#         num_features = train_df.shape[1] - 1  
#         total_train = len(train_df)
#         total_test = len(test_df) if test_df is not None else 0
#         total = total_train + total_test
    
#         def fn_names(fns):
#             if not fns:
#                 return "None"
#             names = []
#             for f in fns:
#                 # Handle functools.partial objects vs regular functions
#                 if hasattr(f, 'func'):
#                     names.append(f.func.__name__)
#                 else:
#                     names.append(f.__name__)
#             return ", ".join(names)
    
#         with open(desc_path, "w") as f:
#             f.write(f"Variant: {self.variant}\n\n")
#             f.write("Dataset Summary\n")
#             f.write("----------------\n")
#             f.write(f"Number of features : {num_features}\n")
#             f.write(f"Total train samples: {total_train}\n")
#             f.write(f"Total test samples : {total_test}\n")
#             f.write(f"Total samples      : {total}\n\n")
#             f.write("Spectogram -> Features\n")
#             f.write("----------------------\n")
#             f.write(f"Pre-processing   : {fn_names(self.pre_process)}\n")
#             f.write(f"Feature extract  : {fn_names(self.features_extract)}\n")
#             f.write(f"Normalization    : {fn_names(self.normalize)}\n")

#     """
#     Pre Process
#     """
#     @staticmethod
#     def resize_specto(specto, shape):
#         # shape (height, width)
#         return resize(specto, shape, anti_aliasing=True, preserve_range=True)

#     @staticmethod
#     def flip_specto(specto):
#         # multiply specto by -1 to change range of dB from 0 to -136 to 0 to 136
#         return -specto

#     """
#     Features Extract
#     """
#     @staticmethod
#     def mean(specto, axis=None):
#         return np.atleast_1d(np.mean(specto, axis))

#     @staticmethod
#     def std(specto, axis=None):
#         return np.atleast_1d(np.std(specto, axis))

#     @staticmethod
#     def median(specto, axis=None):
#         return np.atleast_1d(np.median(specto, axis))

#     @staticmethod
#     def min(specto, axis=None):
#         return np.atleast_1d(np.min(specto, axis))

#     @staticmethod
#     def max(specto, axis=None):
#         return np.atleast_1d(np.max(specto, axis))

#     @staticmethod
#     def minloc(specto, axis=None):
#         return np.atleast_1d(np.argmin(specto, axis))

#     @staticmethod
#     def maxloc(specto, axis=None):
#         return np.atleast_1d(np.argmax(specto, axis))
    
#     """
#     Normalization
#     """
#     @staticmethod
#     def normz(train_df, test_df=None):
#         """
#         Z-score normalize using train statistics.
#         Calculates mean and std for each feature column (all except the last).
#         """
#         if train_df is None:
#             raise ValueError("train_df must not be None for normalization")
#         features_idx = train_df.columns[:-1]
#         train_mean = train_df[features_idx].mean()
#         train_std = train_df[features_idx].std()
#         train_std[train_std == 0] = 1.0
#         new_train = train_df.copy()
#         new_train[features_idx] = (train_df[features_idx] - train_mean) / train_std

#         new_test = None
#         if test_df is not None:
#             new_test = test_df.copy()
#             new_test[features_idx] = (test_df[features_idx] - train_mean) / train_std
#         return new_train, new_test

#     @staticmethod
#     def norm0to1(train_df, test_df=None):
#         """
#         Min-max normalize using train statistics.
#         Scales features to the range [0, 1].
#         """
#         if train_df is None:
#             raise ValueError("train_df must not be None for normalization")
#         features_idx = train_df.columns[:-1]
#         train_min = train_df[features_idx].min()
#         train_max = train_df[features_idx].max()
#         denom = train_max - train_min
#         denom[denom == 0] = 1.0 # Avoid division by zero
#         new_train = train_df.copy()
#         new_train[features_idx] = (train_df[features_idx] - train_min) / denom

#         new_test = None
#         if test_df is not None:
#             new_test = test_df.copy()
#             new_test[features_idx] = (test_df[features_idx] - train_min) / denom
#         return new_train, new_test



    
        



     
        

    
    
        



    