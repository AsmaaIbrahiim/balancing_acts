import torch
import config
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as tt
from model import VAE
from minority_dataset import Minority
from tqdm import tqdm

def save_model(model,optimizer,filename = "model.pth.tar"):
    print("!!!!!! Model Saved Successfully !!!!!!")
    checkpoints = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    torch.save(checkpoints,filename)
trans = tt.Compose([
        tt.Resize((64,64)),
        tt.ToTensor(),
        # tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
dataset = Minority(config.DIR,transform = trans)

val_size = 200
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size,val_size])
train_loader = DataLoader(train_ds,batch_size=config.BATCH_SIZE,shuffle=True)
test_loader = DataLoader(val_ds,batch_size=1,shuffle=False)

if __name__=='__main__':
    model = VAE().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCH):
        loop = tqdm(train_loader,leave=True)
        for idx, data in enumerate(loop):
            data = data.to(config.DEVICE)
            out, pm, logVar = model(data)
            kl_divergence = -0.5 * torch.mean(1+ logVar - pm.pow(2) - logVar.exp())
            loss = F.binary_cross_entropy(out,data,size_average=False) + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('Epoch {}: Loss {}'.format(epoch, loss))

    save_model(model,optimizer,filename="model.pth.tar")