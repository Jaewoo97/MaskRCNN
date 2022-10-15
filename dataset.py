import os
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO


class dropletDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, datatype):
        self.root = root
        self.transforms = transforms
        self.datatype = datatype     # Either HPo, HPi, or all
        self.allANNs = COCO('./data/COCOMasks/all.json')
        # load all image files, sorting them to
        # ensure that they are aligned

        if datatype == 'all':
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))       # Image names
            # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        else:
            tempimgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
            # tempmasks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
            self.imgs = [imglist for imglist in tempimgs if imglist[0:3] == datatype]

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        # mask = np.array(mask)
        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        allMaskListCOCO = [self.allANNs.imgs[ii]['file_name'] for ii in range(0, 50)]
        cat_ids = self.allANNs.getCatIds()
        anns_ids = self.allANNs.getAnnIds(imgIds=allMaskListCOCO.index(self.imgs[idx]), catIds=cat_ids, iscrowd=None)
        anns = self.allANNs.loadAnns(anns_ids)
        maskAnns = anns.copy()
        for i in range(0, len(maskAnns)):
            if maskAnns[i]['area'] < 35.0:
                continue
            if 'masks' in locals():
                masks = np.dstack((masks, self.allANNs.annToMask(maskAnns[i]) )) == 1
            else:
                masks = self.allANNs.annToMask(maskAnns[i]) == 1

        # get bounding box coordinates for each mask
        num_objs = len(masks[0, 0, :])
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)