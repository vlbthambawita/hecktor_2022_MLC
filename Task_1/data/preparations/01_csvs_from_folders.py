import os
import glob
import pandas as pd
import numpy as np


data_dir = "/work/vajira/DATA/Hecktor_2022/resampled_data/training/imagesTr"
mask_dir = "/work/vajira/DATA/Hecktor_2022/resampled_data/training/labelsTr"

csv_path = "/work/vajira/DL/divergent-nets-hecktor/data/all_new_resampled.csv"

img_paths = glob.glob(data_dir + "/*.gz")
mask_paths = glob.glob(mask_dir + "/*.gz")


def make_a_csv(data_dir,  mask_dir, out_csv_path):

    #img_paths = glob.glob(data_dir + "/*.gz")
    mask_paths = glob.glob(mask_dir + "/*.gz")

    with open(csv_path, "w") as f:
        f.writelines("id,ct_path,pt_path,gt_path\n")

        for mask_path in mask_paths:
            code = mask_path.split("/")[-1].split(".")[0]
            ct_path = os.path.join(data_dir, code + "__CT.nii.gz")
            pt_path = os.path.join(data_dir, code + "__PT.nii.gz")
            gt_path = mask_path
            f.write(code)
            f.write(",")
            f.write(ct_path)
            f.write(",")
            f.write(pt_path)
            f.write(",")
            f.write(gt_path)
            f.write("\n")
        f.close()


def divide_train_val(all_csv_file, train_fraction=0.90):
    df = pd.read_csv(all_csv_file)

    train_df, val_df = np.split(df, [int(train_fraction * len(df))])

    train_df.to_csv("new_train_resampled.csv", sep=",", index=False)
    val_df.to_csv("new_val_resampled.csv", sep=",", index=False)

if __name__ == "__main__":

    make_a_csv(data_dir, mask_dir, csv_path)
    divide_train_val(csv_path, train_fraction=0.90)
