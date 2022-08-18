import glob
import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF

class RubberDataset(Dataset):
    def __int__(self, dataRoot, transforms_ = None, transforms_mask = None, subFold="Train_NG", isTrain=True):
        self.isTrain = isTrain
        if transforms_mask == None:
            self.maskTransforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.maskTransforms = transforms_mask

        if transforms_ == None:
            self.transforms = self.maskTransforms
        else:
            self.transforms = transforms_

        self.imgFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.jpg"))

        if isTrain:
            self.labelFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.jpg"))

        self.len = len(self.imgFiles)

    def __getitem__(self, index):
        idx = index % self.len

        if self.isTrain==True:
            img = Image.open(self.imgFiles[idx]).convert("RGB")
            # mask = Image.open(self.labelFiles[idx]).convert("RGB")
            mat = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5, 5), np.uint8)
            matD = cv2.dilate(mat, kernel)
            mask = Image.fromarray(matD)

            if np.random.rand(1) > 0.5:
                mask = VF.hflip(mask)
                img = VF.hflip(img)
            if np.random.rand(1)> 0.5:
                mask = VF.vflip(mask)
                img = VF.vflip(img)

            img = self.transforms(img)
            mask = self.maskTransforms(mask)

            return {"img":img, "mask":mask}
        else:
            img = Image.open(self.imgFiles[idx]).convert("RGB")
            img = self.transforms(img)
            return {"img":img}

    def __len__(self):
        return len(self.imgFiles)