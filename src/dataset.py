import os
import PIL
import torch.utils.data as data


class ImageDataset(data.Dataset): 
    def __init__(self, dataframe, transform=None):
        self.imlist = dataframe 
        self.transform = transform

    def __getitem__(self, index):
        impath, target = self.imlist.loc[index]

        if not os.path.exists(impath): 
            print('No file ', impath)
            raise FileNotFoundError(impath)

        img = PIL.Image.open(impath).convert('RGB')
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imlist)