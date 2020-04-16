import cv2
from PIL import Image
from commons import get_detection_model, get_recognition_model, transform_image

detection_model = get_detection_model()
recognition_model = get_recognition_model()


def get_detection(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = detection_model.forward(tensor)
    except Exception:
         return 0, 'error'
    image = tensor.squeeze(0).mul(255).permute(1, 2, 0).byte().numpy()
    x1, y1, x2, y2 = int(outputs[0]['boxes'][:, 0]), int(outputs[0]['boxes'][:, 1]), int(
        outputs[0]['boxes'][:, 2]), int(outputs[0]['boxes'][:, 3])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    Image.fromarray(image).show()  # 用PIL显示图像
    return a


img = './80.jpg'
a = get_prediction(img)
print(a)
