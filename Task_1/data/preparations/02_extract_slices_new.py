from  PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
#import nibabel as nib
import cv2
import SimpleITK as sitk





def convert_dicom_to_png(input_csv, out_folder):

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "ct_images"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "pt_images"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "masks"), exist_ok=True)

    ct_img_dir = os.path.join(out_folder, "ct_images")
    pt_img_dir = os.path.join(out_folder, "pt_images") 
    mask_dir = os.path.join(out_folder, "masks")

    df = pd.read_csv(input_csv, delimiter=",")

    for i in tqdm(range(len(df))):
        
        row = df.iloc[i]
        # ct_path,pt_path,gt_path
        p_id = row["id"] 
        ct_path = row["ct_path"]
        pt_path = row["pt_path"]
        mask_path = row["gt_path"]
        

        image_ct = sitk.ReadImage(ct_path)
        image_pt =  sitk.ReadImage(pt_path)
        image_mask = sitk.ReadImage(mask_path)

        array_ct = np.transpose(sitk.GetArrayFromImage(image_ct), (2, 1, 0))
        array_pt = np.transpose(sitk.GetArrayFromImage(image_pt), (2, 1, 0))
        array_mask = np.transpose(sitk.GetArrayFromImage(image_mask), (2, 1, 0))

        #print("array_ct_shape", array_ct.shape)
        #print("array_pt_shape", array_pt.shape)
        #print("array_mask_shape", array_mask.shape)





        #nib_data = nib.load(ct_path)
        #np_data = nib_data.get_fdata()

        #nib_mask = nib.load(mask_path)
        #np_mask = nib_mask.get_fdata()
        #print("np_mask unique=", np.unique(np_mask))

    #print(np_data.shape)
        max_pix_ct = np.max(array_ct)
        min_pix_ct = np.min(array_ct)
        max_pix_pt = np.max(array_pt)
        min_pix_pt = np.min(array_pt)

        #max_pix_value_mask = np.max(np_mask)
        #min_pix_value_mask = np.min(np_mask)
    #print("max before=", max_pix_value)
    #print("min before=", min_pix_value)

        scaled_ct = ((array_ct - min_pix_ct)/(max_pix_ct - min_pix_ct)) * 255.0
        scaled_pt = ((array_pt - min_pix_pt)/(max_pix_pt - min_pix_pt)) * 255.0



        scaled_ct = np.uint8(scaled_ct)
        scaled_pt = np.uint8(scaled_pt)
    #print(scaled_data)
        for s in tqdm(range(scaled_ct.shape[2])):
            #img = Image.fromarray(scaled_data[:, :, s], mode="L")
            #img.save(os.path.join(img_dir, "%s_ct_%03d.png" % (p_id,s)))
            cv2.imwrite(os.path.join(ct_img_dir, "%s_ct_%03d.png" % (p_id,s)), scaled_ct[:, :, s])
            cv2.imwrite(os.path.join(pt_img_dir, "%s_pt_%03d.png" % (p_id,s)), scaled_pt[:, :, s])
            cv2.imwrite(os.path.join(mask_dir, "%s_mask_%03d.png" % (p_id,s)), array_mask[:, :, s])
            #pass
            #mask = Image.fromarray(np_mask[:, :, s], mode="L")
            #mask.save(os.path.join(mask_dir, "%s_ct_%03d.png" % (p_id,s)))




        #display(sample_img)



if __name__ == "__main__":

    convert_dicom_to_png("/work/vajira/DL/divergent-nets-hecktor/data/preparations/new_train_resampled.csv", 
    "/work/vajira/DATA/Hecktor_2022/extracted_data/train_resampled")

    convert_dicom_to_png("/work/vajira/DL/divergent-nets-hecktor/data/preparations/new_val_resampled.csv", 
    "/work/vajira/DATA/Hecktor_2022/extracted_data/val_resampled")