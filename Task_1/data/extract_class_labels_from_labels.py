import torch
import nibabel as nib
import numpy as np
import cv2
import glob
import pandas as pd
from tqdm import tqdm


all_files = glob.glob("/work/vajira/DATA/Hecktor_2022/predictions_new_v2" + "/*")

df = pd.DataFrame(columns=["pid", "count_0", "count_1", "count_2"])

for f in tqdm(all_files):
    p_id = f.split("/")[-1].split(".")[0]
    nib_data =  nib.load(f)
    np_data = nib_data.get_fdata()
    dim_z = np_data.shape[2]
    classes, values = np.unique(np_data, return_counts=True)
    #print(classes)
    #print(str(classes))
    row_dict = dict(zip([ "count_" + str(int(c)) for c in classes], values))
    row_dict["pid"] = p_id
    row_dict["dim_z"] = dim_z
    #df_temp = pd.DataFrame(row_dict)
    df  = df.append(row_dict, ignore_index=True).fillna(0)
    #break


df.to_csv("extracted_counts_from_test_data.csv",index=False)