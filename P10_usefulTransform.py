# 常见的图像处理库
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

"""ToTensor"""
# 最终写入的文件夹，writer是SummaryWriter对象，用来将图像写入TensorBoard
writer = SummaryWriter("logs")
# 打开图像
img = Image.open("dataset/train/ants_image/0013035.jpg")
# 这里是创建了一个transforms.ToTensor对象，ToTensor是一个类
# ToTensor里面有__call__，所以可以像函数一样调用它的对象trans_totensor
trans_totensor = transforms.ToTensor()
# 将图像转换为张量
img_tensor = trans_totensor(img)
# 写入TensorBoard
writer.add_image("Tensor_img", img_tensor, 0)

"""Normalize"""
# 对张量进行归一化处理
# 这里与前面的trans_totensor类似，创建了一个transforms.Normalize对象
trans_norms = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
img_norms = trans_norms(img_tensor)
writer.add_image("Normalized_img", img_norms, 0)

"""Resize"""
# 创建一个transforms.Resize对象，指定目标大小为(256, 256)
trans_resize = transforms.Resize((256, 256))
# 对图像进行缩放
img_resize = trans_resize(img)
# 将缩放后的图像转换为张量
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resized_img", img_resize_tensor, 0)

"""Compose-resize-2"""
# Compose本质类似复合函数，这里是先resize再转换tensor
trans_resize_2 = transforms.Resize((256, 256))
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
# 使用Compose对图像进行处理
img_compose = trans_compose(img)
writer.add_image("Compose_Resized_img", img_compose, 0)

writer.close()