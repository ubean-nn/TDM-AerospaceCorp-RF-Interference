def struct_extract(struct, min_label, max_label):
    paths  = []
    labels = []
    for entry in struct:
        label = int(entry[1])
        if min_label <= label <= max_label:
            path = entry[0].split('/')[1].split('.')[0]
            paths.append(path)
            labels.append(label)
    return paths, labels

class HighwayDataset(Dataset):
    def __init__(self, paths, labels, data_root, resize=(64,64)):
        self.paths = paths
        self.labels = labels
        self.data_root = data_root
        self.resize = resize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fname = self.paths[idx] + ".npy"
        arr = np.load(os.path.join(self.data_root, fname))   # shape (512,243)
        arr = np.clip(arr, MIN_DB, MAX_DB) # Clip to a fixed dB range 
        arr = (arr - MIN_DB) / (MAX_DB - MIN_DB) # Scale to [0,1] VERIFY THIS
        img = torch.from_numpy(arr.astype(np.float32))    # H x W
        img = img.unsqueeze(0)                            # 1 Input Channel (amplitude) || 1 x H x W
        img = TF.resize(img, self.resize)                 # RESIZE still 1 x h x w
        label = int(self.labels[idx]) - MIN_LABEL # **** Label shift 
        return img, label