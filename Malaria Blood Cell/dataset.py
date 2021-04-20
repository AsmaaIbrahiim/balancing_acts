import torch
import glob
import config
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tt

class Cell(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        list_files = glob.glob(self.root_dir + "*")
        # print(list_files)
        for class_path in list_files:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "\\*.png"):
                self.data.append([img_path,class_name])
        self.class_map = {"parasitized":0, "uninfected":1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path,class_name = self.data[index]
        img = Image.open(img_path).convert('RGB')
        label = self.class_map[class_name]
        label = torch.tensor([label])
        if self.transform is not None:
            img = self.transform(img)
        return img,label


if __name__=='__main__':
    trans = tt.Compose([
        tt.Resize((128,128)),
        tt.ToTensor()
    ])
    dataset = Cell(config.TRAIN_DIR,transform = trans)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,shuffle=True)
    # print(len(loader))
    for x,y in loader:
        print(x.shape)
        print(y)
        import sys
        sys.exit()