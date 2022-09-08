import os
import glob
import pandas as pd
import numpy as np




def create_csv_from_files(ct_img_dir, pt_img_dir, mask_dir, csv_path):

    #ct_img_paths = glob.glob(ct_img_dir + "/*.png")
    #pt_img_paths = glob.glob(pt_img_dir + "/*.png")
    mask_paths = glob.glob(mask_dir + "/*.png")


    with open(csv_path, "w") as f:
        f.writelines("id,ct_path,pt_path,gt_path\n")

        for mask_path in mask_paths:
            code = mask_path.split("/")[-1].split(".")[0].split("_")[0]
            slice_id = mask_path.split("/")[-1].split(".")[0].split("_")[-1]
            #file_name = mask_path.split("/")[-1]
            ct_path = os.path.join(ct_img_dir, str(code)+ "_ct_" + str(slice_id) + ".png")
            pt_path = os.path.join(pt_img_dir, str(code)+ "_pt_" + str(slice_id) + ".png")
            mask_path = os.path.join(mask_dir, str(code)+ "_mask_" + str(slice_id) + ".png")
            f.write(code)
            f.write(",")
            f.write(ct_path)
            f.write(",")
            f.write(pt_path)
            f.write(",")
            f.write(mask_path)
            f.write("\n")
        f.close()


if __name__ == "__main__":

    pt_dir = "/work/vajira/DATA/Hecktor_2022/extracted_data/val_resampled/pt_images"
    ct_dir  =  "/work/vajira/DATA/Hecktor_2022/extracted_data/val_resampled/ct_images"
    mask_dir = "/work/vajira/DATA/Hecktor_2022/extracted_data/val_resampled/masks"

    csv_path = "/work/vajira/DL/divergent-nets-hecktor/data/val_images_resampled.csv"
    create_csv_from_files(ct_dir, pt_dir, mask_dir, csv_path)