from pathlib import Path
import yaml
import torch

from src.data.dataset import get_dataloaders
from src.models.factory import ModelFactory

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_name = config["model"]["name"]
ckpt_path = Path(config["training"]["save_dir"]) / f"{model_name}_fashion.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelFactory.get_model(
    model_name,
    config["model"]["num_classes"],
    config["model"]["in_channels"]
).to(device)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

_, test_loader = get_dataloaders(batch_size=5, root=config["data"]["root"])
images, labels = next(iter(test_loader))

with torch.no_grad():
    outputs = model(images.to(device))
    preds = outputs.argmax(dim=1)

for true, pred in zip(labels, preds):
    print(f"Реальный: {CLASSES[true]:<12} | Предсказанный: {CLASSES[pred]}")