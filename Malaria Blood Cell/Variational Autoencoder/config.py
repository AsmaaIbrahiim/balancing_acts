import torch

DIR = "E:\\Aquib\\MCA\\Python\\Fatima Fellowship\\datasets\\cell_images\\train\\parasitized\\"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_CHANNELS = 3
FEATURE_DIM = 32*56*56
ZDIM = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCH = 100