import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tt
from torchvision.utils import save_image

TRAIN_DIR = "E:\\Aquib\\MCA\\Python\\Fatima Fellowship\\datasets\\cell_images\\train\\parasitized\\"

class Minority(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_files = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir,img_file)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

if __name__=='__main__':
    trans = tt.Compose([
        tt.Resize((64,64)),
        tt.ToTensor(),
        tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    dataset = Minority(TRAIN_DIR,transform = trans)
    loader = DataLoader(dataset, batch_size=32,shuffle=True)
    for x in loader:
        print(x.shape)
        save_image(x, 'transform.png')
        import sys
        sys.exit()
