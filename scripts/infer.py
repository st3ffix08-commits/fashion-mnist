import sys
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.factory import ModelFactory

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

if len(sys.argv) < 2:
    print("Использование: python scripts/infer.py <путь_к_картинке> [имя_модели]")
    sys.exit(1)

img_path = Path(sys.argv[1])
if not img_path.is_file():
    print(f"Ошибка: файл не найден -> {img_path}")
    sys.exit(1)

model_name = sys.argv[2] if len(sys.argv) > 2 else "resnet18"
ckpt_dir = Path("model_checkpoints") if Path("model_checkpoints").exists() else Path("models_checkpoints")
ckpt_path = ckpt_dir / f"{model_name}_fashion.pth"

if not ckpt_path.is_file():
    print(f"Ошибка: чекпоинт не найден -> {ckpt_path}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor()
])

image = Image.open(img_path)
tensor = transform(image).unsqueeze(0).to(device)
tensor = 1.0 - tensor

model = ModelFactory.get_model(model_name, 10, 1).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

with torch.no_grad():
    output = model(tensor)
    pred_idx = output.argmax(dim=1).item()

predicted_class = CLASSES[pred_idx]
print(f"Класс: {predicted_class}")

# Визуализация того, что пришло на вход модели
vis_tensor = tensor.cpu().squeeze() # Убираем батч и канал -> (28, 28)

plt.figure(figsize=(4, 4))
plt.imshow(vis_tensor, cmap="gray")
plt.title(f"Pred: {predicted_class}")
plt.axis("off")
plt.show()