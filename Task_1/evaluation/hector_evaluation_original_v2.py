from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pandas as pd
import os
import glob



#prediction_folder = '../data/hecktor2022_testing/predictions'
##groundtruth_folder = '../data/hecktor2022_testing/labelsTs'

# List of the files in the validation
#prediction_files = [f for f in Path(prediction_folder).rglob('*.nii.gz')]

# The list is sorted, so it will match the list of ground truth files
##prediction_files.sort(key=lambda x: x.name.split('.')[0])

# List of the patient_id in the validation
#patient_name_predictions = [f.name.split('.')[0] for f in prediction_files]


# List of the ground truth files
#groundtruth_files = [
#    f for f in Path(groundtruth_folder).rglob('*.nii.gz') if f.name.split('.')[0] in patient_name_predictions
#]

#df = pd.read_csv("/work/vajira/DL/divergent-nets-hecktor/data/preparations/new_val_resampled.csv")

original_mask_dir  = "/work/vajira/DATA/Hecktor_2022/original_data/hecktor2022_training/hecktor2022/labelsTr" #"/work/vajira/DATA/Hecktor_2022/resampled_data/training/labelsTr"#"/work/vajira/DATA/Hecktor_2022/original_data/hecktor2022_training/hecktor2022/labelsTr"
prediction_dir = "/work/vajira/DATA/Hecktor_2022/submission/only_CT_validation_set"

patient_names = [pid.split("/")[-1].split(".")[0] for pid in glob.glob(prediction_dir + "/*.nii.gz")]

#print(patient_names)



def compute_volumes(im):
    """
    Compute the volumes of the GTVp and the GTVn
    """
    spacing = im.GetSpacing()
    voxvol = spacing[0] * spacing[1] * spacing[2]
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(im, im)
    nvoxels1 = stats.GetCount(1)
    nvoxels2 = stats.GetCount(2)
    return nvoxels1 * voxvol, nvoxels2 * voxvol


def compute_agg_dice(intermediate_results):
    """
    Compute the aggregate dice score from the intermediate results
    """
    aggregate_results = {}
    TP1s = [v["TP1"] for v in intermediate_results]
    TP2s = [v["TP2"] for v in intermediate_results]
    vol_sum1s = [v["vol_sum1"] for v in intermediate_results]
    vol_sum2s = [v["vol_sum2"] for v in intermediate_results]
    DSCagg1 = 2 * np.sum(TP1s) / np.sum(vol_sum1s)
    DSCagg2 = 2 * np.sum(TP2s) / np.sum(vol_sum2s)
    aggregate_results['AggregatedDsc'] = {
        'GTVp': DSCagg1,
        'GTVn': DSCagg2,
        'mean': np.mean((DSCagg1, DSCagg2)),
    }

    return aggregate_results

def get_intermediate_metrics(groundtruth, prediction):
    """
    Compute intermediate metrics for a given groundtruth and prediction.
    These metrics are used to compute the aggregate dice.
    """
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(groundtruth, prediction)

    DSC1 = overlap_measures.GetDiceCoefficient(1)
    DSC2 = overlap_measures.GetDiceCoefficient(2)

    vol_gt1, vol_gt2 = compute_volumes(groundtruth)
    vol_pred1, vol_pred2 = compute_volumes(prediction)

    vol_sum1 = vol_gt1 + vol_pred1
    vol_sum2 = vol_gt2 + vol_pred2
    TP1 = DSC1 * (vol_sum1) / 2
    TP2 = DSC2 * (vol_sum2) / 2
    return {
        "TP1": TP1,
        "TP2": TP2,
        "vol_sum1": vol_sum1,
        "vol_sum2": vol_sum2,
    }

def resample_prediction(groundtruth, prediction):
    """
    Resample the prediction to the groundtruth physical domain
    """
    resample = sitk.ResampleImageFilter()
    resample.SetSize(groundtruth.GetSize())
    resample.SetOutputDirection(groundtruth.GetDirection())
    resample.SetOutputOrigin(groundtruth.GetOrigin())
    resample.SetOutputSpacing(groundtruth.GetSpacing())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(prediction) 

def check_prediction(groundtruth, prediction):
    """
    Check if the prediction is valid and apply padding if needed
    """

    # Cast to the same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    groundtruth = caster.Execute(groundtruth)
    prediction = caster.Execute(prediction)

    # Check labels
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(prediction, prediction)
    labels = stats.GetLabels()
    if not all([l in [0, 1, 2] for l in labels]):
        raise RuntimeError(
            "The labels are incorrect. The labels should be background: 0, GTVp: 1, GTVn: 2."
        )
    # Check spacings
    if not np.allclose(
            groundtruth.GetSpacing(), prediction.GetSpacing(), atol=0.000001):
        raise RuntimeError(
            "The resolution of the prediction is different from the CT resolution."
        )
    else:
        # to be sure that sitk won't trigger unnecessary errors
        prediction.SetSpacing(groundtruth.GetSpacing())

    # the resample_prediction is used to crop the prediction to the same size as the groundtruth
    return resample_prediction(groundtruth, prediction)

results = list()
for patient_name in patient_names:
    #patient_name = row["id"]#f.name.split('.')[0]
    #gt_file = [k for k in groundtruth_files if k.name[:7] == patient_name][0]

    print(f"Evaluating patient {patient_name}")

    prediction = sitk.ReadImage(os.path.join(prediction_dir, patient_name + ".nii.gz"))
    groundtruth = sitk.ReadImage(os.path.join(original_mask_dir, patient_name + ".nii.gz"))
    prediction = check_prediction(groundtruth, prediction) 

    results.append(get_intermediate_metrics(groundtruth, prediction))

print("The results are:")
print(compute_agg_dice(results))
