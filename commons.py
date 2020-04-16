import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms


def get_detection_model():
    model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91,
                                                              pretrained=True)
    model.to('cpu')
    checkpoint = torch.load('./water_meter_detection.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor()])
    # image = Image.open(io.BytesIO(image_bytes))
    image = Image.open(image_bytes)
    return my_transforms(image).unsqueeze(0)


