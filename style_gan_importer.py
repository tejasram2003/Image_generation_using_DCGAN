import torch
from pytorch_pretrained_gans import make_gan
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import tqdm
import random

G = make_gan(gan_type='selfconditionedgan', model_name='self_conditioned')

y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])

image = x.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()

num_epochs = 100

# for epoch in range(num_epochs):
#     print(f"epoch: {epoch}, loss: {random.randint(0,100)}")
#     for i in tqdm.tqdm(range(100)):
#         val = i
#     time.sleep(5)