from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
writer.add_image("Tensor_img", img_tensor, 0)
writer.close()