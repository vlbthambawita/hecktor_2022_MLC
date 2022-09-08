from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import copy


image_folder_original = Path("/work/vajira/DATA/Hecktor_2022/original_data/hecktor2022_testing/imagesTs").resolve()
image_folder_resampled = Path("/work/vajira/DATA/Hecktor_2022/resampled_data/testing/imagesTs").resolve()
#image_folder_original = Path("/work/vajira/DATA/Hecktor_2022/original_data/hecktor2022_testing/imagesTs").resolve()
#image_folder_resampled = Path("/work/vajira/DATA/Hecktor_2022/resampled_data/testing/imagesTs").resolve()
results_folder = Path("/work/vajira/DATA/Hecktor_2022/submission/test_resampled_submission_new").resolve()
results_folder.mkdir(exist_ok=True)

patient_list = [f.name.split("__")[0] for f in image_folder_original.rglob("*__CT.nii.gz")]


post_resampler = sitk.ResampleImageFilter()
post_resampler.SetInterpolator(sitk.sitkNearestNeighbor)

resampling_spacing = np.array([2.0, 2.0, 2.0]) # from resampling script

for patient_id in tqdm(patient_list):
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
    dummy_prediction = copy.copy(array_ct)
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
