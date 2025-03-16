from PIL import ImageOps
from retinaface import RetinaFace
from torchvision import transforms
import torch
import numpy as np

def detect_and_crop_face(image):
    """Detects a face in an image, crops it, and resizes it to 224x224."""
    # Convert PIL Image (grayscale) to RGB
    image = image.convert('RGB')

    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    faces = RetinaFace.detect_faces(image_np)

    if len(faces) >= 2:
        return "Mutiple Faces Detected.."
    elif len(faces) == 0:
        return "No Face Detected.."

    # print(faces)
    key = list(faces.keys())[0]
    face = faces[key]
    x1, y1, x2, y2 = face["facial_area"]

    # Expand bounding box
    expansion = 60
    x1 = max(0, x1 - expansion)
    y1 = max(0, y1 - expansion)
    x2 = min(image.size[1], x2 + expansion)
    y2 = min(image.size[0], y2 + expansion)

    # Crop and resize face
    face_crop = image.crop((x1, y1, x2, y2))
    face_resized = face_crop.resize((224, 224))

    return face_resized


def preprocessing(image):
    """Preprocesses an image for classification."""
    transformation = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])
    image = transformation(image)
    image = image.unsqueeze(0)
    return image


def classifier(image, model, class_names):
    
    preprocessed_img = detect_and_crop_face(image)
    
    if type(preprocessed_img) == str:
        return preprocessed_img, "Error"
    
    
    preprocessed_img = ImageOps.grayscale(preprocessed_img)
    preprocessed_img = preprocessing(preprocessed_img)

    output = model(preprocessed_img)
    pred = torch.argmax(output).item()
    label = class_names[pred]
    score = output[0][pred].item()
    score = round(score * 100, 2)
    score = int(score)
    
    return label,score