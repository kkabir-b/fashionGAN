#used to store useful functions
import numpy as np
import matplotlib.pyplot as plt

def save_images(loader):
    real_batch = next(iter(loader))
    image = real_batch[0][0]
    np_image = image.numpy().squeeze()
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np_image,cmap = 'gray')
    plt.savefig('test.png')

