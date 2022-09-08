
import pandas as pd
import albumentations as albu
from torch.utils.data import DataLoader

from data.dataset import DatasetImageV2 as Dataset

def df_from_csv_file_array(csv_file_arrya):

    df =pd.DataFrame(columns=["image", "path"])

    for csv in csv_file_arrya:
        temp_df = pd.read_csv(csv)

        df = df.append(temp_df)

    return df



def get_training_augmentation(opt):
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=opt.img_size, min_width=opt.img_size, always_apply=True, border_mode=0),
        albu.Resize(height=opt.img_size, width=opt.img_size, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                #albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)



def get_validation_augmentation(opt):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=opt.img_size, min_width=opt.img_size, always_apply=True, border_mode=0),
        albu.Resize(height=opt.img_size, width=opt.img_size, always_apply=True),
    ]
    return albu.Compose(test_transform)



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = []
    if preprocessing_fn:
        _transform.append(albu.Lambda(image=preprocessing_fn))
    _transform.append(albu.Lambda(image=to_tensor, mask=to_tensor))

    
    return albu.Compose(_transform)



def prepare_data(opt, preprocessing_fn):


    
    train_df = df_from_csv_file_array(opt.train_CSVs)
    val_df = df_from_csv_file_array(opt.val_CSVs)


    train_dataset = Dataset(
        train_df,
        grid_sizes=opt.grid_sizes_train,
        augmentation=get_training_augmentation(opt), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )

    valid_dataset = Dataset(
        val_df, 
        grid_sizes=opt.grid_sizes_val,
        augmentation=get_validation_augmentation(opt), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=4)

    
   
    print("dataset train=", len(train_dataset))
    print("dataset val=", len(valid_dataset))

    return train_loader, valid_loader

def prepare_test_data(opt, preprocessing_fn):

    test_df = df_from_csv_file_array(opt.test_CSVs)

    # test dataset without transformations for image visualization
    test_dataset = Dataset(
        test_df,
        grid_sizes=opt.grid_sizes_test,
        augmentation=get_validation_augmentation(opt), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )

    print("Test dataset size=", len(test_dataset))

    return test_dataset
