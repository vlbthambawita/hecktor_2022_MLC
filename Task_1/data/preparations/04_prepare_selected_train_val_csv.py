import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm



# df = pd.read_csv("/work/vajira/DL/divergent-nets-hecktor/data/train_images_new.csv")


def prepare_selected_train_csvs(csv_path):
    df = pd.read_csv(csv_path)
    df_selected = pd.DataFrame(columns=["id","ct_path","pt_path", "gt_path"])
    df_blank = pd.DataFrame(columns=["id","ct_path","pt_path","gt_path"])


    for i, r in tqdm(df.iterrows()):
        #print(r["gt_path"])
        gt_path = r["gt_path"]
        mask = cv2.imread(gt_path)
        if len(np.unique(mask)) > 1:
            df_selected = df_selected.append({"id": r["id"], "ct_path": r["ct_path"], "pt_path": r["pt_path"], "gt_path": r["gt_path"] }, ignore_index=True)
        else:
            df_blank = df_blank.append({"id": r["id"], "ct_path": r["ct_path"], "pt_path": r["pt_path"], "gt_path": r["gt_path"] }, ignore_index=True)

    df_blank_selected = df_blank.sample(n=len(df_selected))
    
    df_selected.to_csv("train_images_with_multiclass_resampled.csv", index=False)
    df_blank.to_csv("train_images_blank_resampled.csv", index=False)
    df_blank_selected.to_csv("train_images_selected_blank_resampled.csv", index=False)
    

def prepare_selected_test_csv(csv_path):

    df = pd.read_csv(csv_path)

    df_selected = pd.DataFrame(columns=["id","ct_path","pt_path","gt_path"])
    for i, r in tqdm(df.iterrows()):
        #print(r["gt_path"])
        gt_path = r["gt_path"]
        mask = cv2.imread(gt_path)
        if len(np.unique(mask)) > 1:
            df_selected = df_selected.append({"id": r["id"], "ct_path": r["ct_path"],"pt_path": r["pt_path"], "gt_path": r["gt_path"] }, ignore_index=True)

    df_selected.to_csv("test_images_with_only_multiclass_resampled.csv", index=False)

if __name__== "__main__":

    #prepare_selected_train_csvs("/work/vajira/DL/divergent-nets-hecktor/data/train_images_resampled.csv")
    prepare_selected_test_csv("/work/vajira/DL/divergent-nets-hecktor/data/val_images_resampled.csv")