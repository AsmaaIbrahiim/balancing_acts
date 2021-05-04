import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import torchvision
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from minority_dataset import Minority

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "E:\\Aquib\\MCA\\Python\\Fatima Fellowship\\datasets\\cell_images\\train\\parasitized\\"
results = 'E:\\Aquib\\MCA\\Python\\Fatima Fellowship\\DCGAN\\results\\'
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

trans = tt.Compose([
    tt.Resize((64,64)),
    tt.ToTensor(),
    tt.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
dataset = Minority(TRAIN_DIR,transform = trans)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

gen = Generator(NOISE_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE,NOISE_DIM,1,1).to(device)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 20 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                save_image(fake*0.5 + 0.5, results+f"fake{epoch}.png")

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1