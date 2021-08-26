import h5py
from PIL import Image, ImageOps
from glob import glob
import numpy as np
import os
from tqdm import tqdm

data_path = r"E:/dataset/mirflickr"
files_list = glob(os.path.join(data_path, '*.jpg'))
shape = (len(files_list), 400, 400, 3)

hdf5_path = r"../data/images.hdf5"
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("image", shape, float)

for i, file in enumerate(tqdm(files_list)):
    img_cover = Image.open(file).convert('RGB')
    img_cover = ImageOps.fit(img_cover, (shape[1], shape[2]))
    img_cover = np.array(img_cover) / 255
    hdf5_file["image"][i] = img_cover

hdf5_file.close()
