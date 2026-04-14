import numpy as np
import os

files
file_path = "path/to/your/data_file.npy" 

print(f"Inspecting: {file_path}")

try:
    # 1. Attempt to load based on extension
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)['arr_0'] # Usually 'arr_0' or inspect keys
    elif file_path.endswith('.mat'):
        import scipy.io
        mat = scipy.io.loadmat(file_path)
        # MAT files are dictionaries, we need to find the data key
        print("Keys in MAT file:", mat.keys())
        key = [k for k in mat.keys() if not k.startswith('_')][0] # Grab first data key
        data = mat[key]
    elif file_path.endswith(('.h5', '.hdf5')):
        import h5py
        with h5py.File(file_path, 'r') as f:
            print("Keys in H5 file:", list(f.keys()))
            key = list(f.keys())[0]
            data = f[key][:] # Load first dataset into memory
    else:
        # Fallback for raw binary (assuming float32 - adjust if needed)
        data = np.fromfile(file_path, dtype=np.float32)

    # 2. Print the Critical Stats
    print("-" * 30)
    print(f"SHAPE: {data.shape}")
    print(f"DTYPE: {data.dtype}")
    print(f"MIN/MAX: {np.min(data)} / {np.max(data)}")
    
    # 3. Check for Complex Numbers (I/Q Data)
    if np.iscomplexobj(data):
        print("TYPE: Complex-Valued (I/Q Data) detected!")
    else:
        print("TYPE: Real-Valued Data")

    print("-" * 30)
    
    # 4. Preview the first sample (if possible)
    if data.ndim > 1:
        print("First sample shape:", data[0].shape)
    else:
        print("Data is 1D array.")

except Exception as e:
    print(f"Error loading file: {e}")