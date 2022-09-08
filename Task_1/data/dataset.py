from torch.utils.data import Dataset as BaseDataset
import pyra_pytorch
import numpy as np

import nibabel as nib
import pandas as pd
from PIL import Image
import cv2


class DatasetOld(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            df, 
            classes=[0,255], 
            grid_sizes = [256],
            augmentation=None, 
            preprocessing=None,
            pyra=False
    ):
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.grid_sizes = grid_sizes

        self.pyra_dataset = pyra_pytorch.PYRADatasetFromDF(df, grid_sizes=grid_sizes)
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.pyra = pyra
    
    def __getitem__(self, i):
        
        # read data
        data = self.pyra_dataset[i]
        #image = cv2.imread(self.images_fps[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i], 0)
        image = data["img"]
        grid = np.expand_dims(data["grid_encode"], axis=2)
        mask = data["mask"]
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.pyra:
            image = np.concatenate([image, grid], axis=2)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.pyra_dataset)



class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            df, 
            classes=[0,1,2], 
            augmentation=None, 
            preprocessing=None,
            grid_sizes=256,
            pyra=None
    ):
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.grid_sizes = grid_sizes
        self.pyra = pyra

        #self.pyra_dataset = pyra_pytorch.PYRADatasetFromDF(df, grid_sizes=grid_sizes)
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.df = df
    
    def __getitem__(self, i):
        
        # read data
        data_paths = self.df.iloc[i]
        # ct_path,pt_path,gt_path
        ct_path = data_paths["ct_path"]
        pt_path = data_paths["pt_path"]

        mask_path = data_paths["gt_path"]

        ct_3d_data =  nib.load(ct_path)
        #pt_3d_data = nib.load(pt_path)
        mask_data = nib.load(mask_path)

        ct_3d_data = ct_3d_data.get_fdata()
        #pt_3d_data = pt_3d_data.get_fdata()
        mask_data = mask_data.get_fdata()

        #print("ct=", ct_3d_data.shape)
        #print("pt=", pt_3d_data.shape)
        #print("mask=", mask_data.shape)


        ran_slice_num = np.random.randint(0, ct_3d_data.shape[2]-1)
        #print("ran slice num=", ran_slice_num)

        #image = np.array(Image.open(img_path))
        #mask = np.array(Image.open(mask_path))
        ct_slice = np.array(ct_3d_data[:, :, 0:1])
        #pt_slice = np.array(pt_3d_data[:, :, ran_slice_num])
    
        mask_slice =  np.array(mask_data[:, :, ran_slice_num])

        #ct_slice = np.expand_dims(ct_slice, axis=0)
        #ct_slice = np.tra
        

        mask = mask_slice
        image = ct_slice


        #print("image shape=",image.shape)
        #print("mask shape=", mask.shape)

        #print(np.unique(image))
        #print("mask=",np.unique(mask))
        #print("mask one =",np.unique(mask[:,:, 0]))
        #mask = mask[:,:, 0]
        #print("befire=", np.unique(mask))
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # change 0,1,2,3,4,5 range into 0-255
        #mask = mask * 255
        #print(mask.shape)
        #print(np.unique(mask[:,:,0]))
        #print(np.unique(mask[:,:,1]))
        #print(np.unique(mask[:,:,2]))
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #if self.pyra:
        #    image = np.concatenate([image, grid], axis=2)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
       # print("shap ct slicee=", image.shape)
       # print("shape mask=", mask.shape)
       # print("ct slice type=", type(image))
       # print("mask type=", type(mask))
        return image,  mask #pt_slice, mask
        
    def __len__(self):
        return len(self.df)


class DatasetImage(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            df, 
            classes=[0,1,2], 
            augmentation=None, 
            preprocessing=None,
            grid_sizes=256,
            pyra=None
    ):
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.grid_sizes = grid_sizes
        self.pyra = pyra

        #self.pyra_dataset = pyra_pytorch.PYRADatasetFromDF(df, grid_sizes=grid_sizes)
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.df = df
    
    def __getitem__(self, i):
        
        # read data
        data_paths = self.df.iloc[i]
        # ct_path,pt_path,gt_path
        img_path = data_paths["ct_path"]
        #pt_path = data_paths["pt_path"]

        mask_path = data_paths["gt_path"]
        #print("img path=", img_path)
        #print("mask path=", mask_path)

        #image = cv2.imread(img_path)
        #mask = cv2.imread(mask_path)
        image = np.array(Image.open(img_path))
        image = np.expand_dims(image, axis=2)

        mask = np.array(Image.open(mask_path))

        #ct_3d_data =  nib.load(ct_path)
        #pt_3d_data = nib.load(pt_path)
        #mask_data = nib.load(mask_path)

        #ct_3d_data = ct_3d_data.get_fdata()
        #pt_3d_data = pt_3d_data.get_fdata()
        #mask_data = mask_data.get_fdata()

        #print("ct=", ct_3d_data.shape)
        #print("pt=", pt_3d_data.shape)
        #print("mask=", mask_data.shape)


        #ran_slice_num = np.random.randint(0, ct_3d_data.shape[2]-1)
        #print("ran slice num=", ran_slice_num)

        #image = np.array(Image.open(img_path))
        #mask = np.array(Image.open(mask_path))
        #ct_slice = np.array(ct_3d_data[:, :, 0:1])
        #pt_slice = np.array(pt_3d_data[:, :, ran_slice_num])
    
        #mask_slice =  np.array(mask_data[:, :, ran_slice_num])

        #ct_slice = np.expand_dims(ct_slice, axis=0)
        #ct_slice = np.tra
        

        #mask = mask_slice
        #image = ct_slice


        #print("image shape=",image.shape)
        #print("mask shape=", mask.shape)

        #print(np.unique(image))
        #print("mask=",np.unique(mask))
        #print("mask one =",np.unique(mask[:,:, 0]))
        #mask = mask[:,:, 0]
        #print("befire=", np.unique(mask))
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # change 0,1,2,3,4,5 range into 0-255
        #mask = mask * 255
        #print(mask.shape)
        #print(np.unique(mask[:,:,0]))
        #print(np.unique(mask[:,:,1]))
        #print(np.unique(mask[:,:,2]))
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #if self.pyra:
        #    image = np.concatenate([image, grid], axis=2)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        #print("shap gt=", image.shape)
        #print("shape mask=", mask.shape)
       # print("ct slice type=", type(image))
       # print("mask type=", type(mask))
        return image,  mask #pt_slice, mask
        
    def __len__(self):
        return len(self.df)

class DatasetImageV2(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            df, 
            classes=[0,1,2], 
            augmentation=None, 
            preprocessing=None,
            grid_sizes=256,
            pyra=None
    ):
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.grid_sizes = grid_sizes
        self.pyra = pyra

        #self.pyra_dataset = pyra_pytorch.PYRADatasetFromDF(df, grid_sizes=grid_sizes)
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.df = df
    
    def __getitem__(self, i):
        
        # read data
        data_paths = self.df.iloc[i]
        # ct_path,pt_path,gt_path
        ct_path = data_paths["ct_path"]
        pt_path = data_paths["pt_path"]

        mask_path = data_paths["gt_path"]
        #print("img path=", img_path)
        #print("mask path=", mask_path)

        #image = cv2.imread(img_path)
        #mask = cv2.imread(mask_path)
        ct_image = np.array(Image.open(ct_path))
        pt_image = np.array(Image.open(pt_path))
        mask = np.array(Image.open(mask_path))

        ct_pt_mean = (ct_image + pt_image)/2
        ct_pt_mean = np.uint8(np.round(ct_pt_mean))
        #print("ct_pt_mean_shape=", ct_pt_mean.shape)

        ct_pt_image = np.stack([ct_image, pt_image, ct_pt_mean], axis=2)
        image = ct_pt_image

        #print("ct_image_shape=", ct_image.shape)
        #print("pt_image_shape=", pt_image.shape)
        #print("mask shape=", mask.shape)
        #print("ct_pt_imag_shape=", ct_pt_image.shape)

       
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # change 0,1,2,3,4,5 range into 0-255
        #mask = mask * 255
        #print(mask.shape)
        #print(np.unique(mask[:,:,0]))
        #print(np.unique(mask[:,:,1]))
        #print(np.unique(mask[:,:,2]))
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #if self.pyra:
        #    image = np.concatenate([image, grid], axis=2)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        #print("shap gt=", image.shape)
        #print("shape mask=", mask.shape)
       # print("ct slice type=", type(image))
       # print("mask type=", type(mask))
        #print("image shape", image.shape)
        return image,  mask #pt_slice, mask
        
    def __len__(self):
        return len(self.df)

if __name__=="__main__":

    df = pd.read_csv("/work/vajira/DL/divergent-nets-hecktor/data/preparations/train_images_with_multiclass_resampled.csv", delimiter=",")
    dataset = DatasetImageV2(df)

    print(len(dataset))
    test =dataset[2000]
