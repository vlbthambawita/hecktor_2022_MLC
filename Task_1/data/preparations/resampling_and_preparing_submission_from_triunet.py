from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import copy
import pandas as pd
import cv2
import torch


image_folder_original = Path("/work/vajira/DATA/Hecktor_2022/original_data/hecktor2022_testing/imagesTs").resolve()
image_folder_resampled = Path("/work/vajira/DATA/Hecktor_2022/resampled_data/testing/imagesTs").resolve()
#image_folder_original = Path("/work/vajira/DATA/Hecktor_2022/original_data/hecktor2022_testing/imagesTs").resolve()
#image_folder_resampled = Path("/work/vajira/DATA/Hecktor_2022/resampled_data/testing/imagesTs").resolve()
results_folder = Path("/work/vajira/DATA/Hecktor_2022/submission/004_triUnet_test_submission_final").resolve() #predictions_val_old_submission
results_folder.mkdir(exist_ok=True)

patient_list = [f.name.split("__")[0] for f in image_folder_original.rglob("*__CT.nii.gz")]

#df = pd.read_csv("/work/vajira/DL/divergent-nets-hecktor/data/preparations/new_val_resampled.csv")

torch.cuda.set_device(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_CHECK_POINT_PATH = "/work/vajira/DATA/Hecktor_2022/checkpoint/004_tri_unet_image_resampled_images_wandb.py/checkpoints/best_checkpoint.pth"

#submission_dir = "/work/vajira/DATA/Hecktor_2022/submission"
#out_dir = "/work/vajira/DATA/Hecktor_2022/predictions_new_v2"#
chkpt = torch.load(BEST_CHECK_POINT_PATH, map_location=DEVICE)
print("Checkpoint epoch=", chkpt["epoch"])
model = chkpt["model"]

def predict(array_ct, array_pt):

    max_ct = np.max(array_ct)
    min_ct = np.min(array_ct)
    max_pt = np.max(array_pt)
    min_pt = np.min(array_pt)

    scaled_ct = ((array_ct - min_ct)/(max_ct - min_ct)) * 255.0
    scaled_pt = ((array_pt - min_pt)/(max_pt - min_pt)) * 255.0

    all_pred = []

    for slice_id in range(scaled_ct.shape[2]):
        slice_ct= scaled_ct[:,:, slice_id]
        slice_pt= scaled_pt[:,:, slice_id]

        slice_ct_pt_mean = (slice_ct + slice_pt)/2
        slice_ct_pt_mean = np.uint8(np.round(slice_ct_pt_mean))
        #print("ct_pt_mean_shape=", ct_pt_mean.shape)

        slice_ct_pt_input = np.stack([slice_ct, slice_pt, slice_ct_pt_mean], axis=2)
        

        res_slice = cv2.resize(slice_ct_pt_input, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        res_slice = np.transpose(res_slice, (2, 0, 1))
        #print(res_slice.shape)
        input_slice = torch.from_numpy(res_slice)
        input_slice = input_slice.unsqueeze(0)
        input_slice = input_slice.type(torch.FloatTensor).to(device=DEVICE)
        
        # do predictions
        pred =model.predict(input_slice)
        pred_round = torch.round(pred)

        bg = pred_round[0, 0, :,:]
        gtvp = pred_round[0, 1, :,:] * 2
        gtvn = pred_round[0, 2, :, :] * 3 

        pred_stacked = torch.stack([bg, gtvp, gtvn])
        pred_max = torch.max(pred_stacked, dim=0).values

        pred_max_matched  = torch.clamp(pred_max - torch.tensor(1, device=DEVICE).expand_as(pred_max), min=0,max=2) # to match with given class labels 

        pred_max_cpu = pred_max_matched.cpu().numpy()
        #pred_max_cpu_resized = cv2.resize(pred_max_cpu, dsize=(512, 512))
        pred_max_cpu_resized = np.rint(pred_max_cpu)
        all_pred.append(pred_max_cpu_resized)


        #print(pred_max_cpu_resized.shape)
    final_pred = np.stack(all_pred, axis=2)
    
    return final_pred #numpy array x, y , z



post_resampler = sitk.ResampleImageFilter()
post_resampler.SetInterpolator(sitk.sitkNearestNeighbor)

resampling_spacing = np.array([2.0, 2.0, 2.0]) # from resampling script

for p_id in tqdm(patient_list):
    patient_id = p_id
    # loading the images and storing the ct spacing
    image_ct_original = sitk.ReadImage(str(image_folder_original / (patient_id + "__CT.nii.gz")))
    image_ct = sitk.ReadImage(str(image_folder_resampled / (patient_id + "__CT.nii.gz")))
    image_pt = sitk.ReadImage(str(image_folder_resampled / (patient_id + "__PT.nii.gz")))


    # sitk to numpy, sitk stores images with [dim_z, dim_y, dim_x]
    array_ct_original = np.transpose(sitk.GetArrayFromImage(image_ct_original), (2, 1, 0))
    array_ct = np.transpose(sitk.GetArrayFromImage(image_ct), (2, 1, 0))
    array_pt = np.transpose(sitk.GetArrayFromImage(image_pt), (2, 1, 0))

    # original attributed
    ct_origin_resampled = image_ct.GetOrigin() # origin of resampled data
    ct_origin_space = image_ct_original.GetSpacing()


    #  Prediction opereations goes here
    dummy_prediction =predict(array_ct, array_pt)
    #print("prediction shape=", dummy_prediction.shape)


    #dummy_prediction = copy.copy(array_ct)
    image_dummy_prediction = sitk.GetImageFromArray(np.transpose(dummy_prediction, (2, 1, 0)))
    image_dummy_prediction.SetOrigin(ct_origin_resampled) #??
    image_dummy_prediction.SetSpacing(resampling_spacing)  #??
    # End of predictions


    size_ct = image_ct_original.GetSize()
    direction_ct = image_ct_original.GetDirection()
    origin_ct = image_ct_original.GetOrigin()
    spacing_ct = image_ct_original.GetSpacing()
    # Resample to the original CT resolution.
    # You are welcomed to use any fancier interpolation here.
    post_resampler.SetSize(size_ct)
    post_resampler.SetOutputDirection(direction_ct)
    post_resampler.SetOutputOrigin(origin_ct)
    post_resampler.SetOutputSpacing(spacing_ct)
    image_segmentation = post_resampler.Execute(image_dummy_prediction) 


    image_segmentation = post_resampler.Execute(image_segmentation)

    final_array  = np.transpose(sitk.GetArrayFromImage(image_segmentation), (2, 1, 0))


    print("orginal ct size=", array_ct_original.shape)
    print("resampled ct size =", array_ct.shape)
    print("resampled pt size=", array_pt.shape)
    print("final size=", final_array.shape)

    # Saving the prediction
    sitk.WriteImage(
        image_segmentation,
        str(results_folder / (patient_id + ".nii.gz")),
    )


