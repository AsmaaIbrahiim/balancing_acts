import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = 'E:\\Aquib\\MCA\\Python\\Fatima Fellowship\\DCGAN\\generated_images\\'
LEARNING_RATE = 2e-4
NOISE_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 64
CHECKPOINT_GEN = "E:\\Aquib\\MCA\\Python\\Fatima Fellowship\generator.pth.tar"


gen = Generator(NOISE_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))


def load_model(checkpoint_file,model,optimizer,lr):
    print("!!!!!! Loading Model !!!!!!")
    checkpoint =torch.load(checkpoint_file,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

load_model(checkpoint_file=CHECKPOINT_GEN,model=gen,optimizer=opt_gen,lr=LEARNING_RATE)

for i in range(1,11480):
    fixed_noise = torch.randn(1,NOISE_DIM,1,1).to(device)
    gen.eval()
    with torch.no_grad():
        fake = gen(fixed_noise)
        save_image(fake*0.5 + 0.5, results+f"fake{i}.png")