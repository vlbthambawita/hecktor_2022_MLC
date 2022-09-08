import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm



df = pd.read_csv("/work/vajira/DL/divergent-nets-hecktor/data/train_images_new.csv")
df_selected = pd.DataFrame(columns=["id","ct_path","gt_path"])
df_blank = pd.DataFrame(columns=["id","ct_path","gt_path"])


for i, r in tqdm(df.iterrows()):
    #print(r["gt_path"])
    gt_path = r["gt_path"]
    mask = cv2.imread(gt_path)
    if len(np.unique(mask)) > 1:
        df_selected = df_selected.append({"id": r["id"], "ct_path": r["ct_path"], "gt_path": r["gt_path"] }, ignore_index=True)
    else:
        df_blank = df_blank.append({"id": r["id"], "ct_path": r["ct_path"], "gt_path": r["gt_path"] }, ignore_index=True)

df_selected.to_csv("train_images_with_multiclass.csv", index=False)
df_blank.to_csv("train_images_blank.csv", index=False)