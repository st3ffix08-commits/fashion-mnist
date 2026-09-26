from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

from src.data.dataset import get_dataloaders
from src.models.factory import ModelFactory
from src.utils.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = get_dataloaders(batch_size=64, root="./data/raw")
criterion = nn.CrossEntropyLoss()

models = ["resnet18", "vgg11", "alexnet"]
results = []

for name in models:
    ckpt_path = Path("model_checkpoints") / f"{name}_fashion.pth"
    if not ckpt_path.exists():
        continue
    
    model = ModelFactory.get_model(name, num_classes=10, in_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    trainer = Trainer(model=model, criterion=criterion, optimizer=None, device=device)
    loss, acc = trainer.evaluate(test_loader)
    
    results.append({
        "Модель": name,
        "Test Loss": round(loss, 4),
        "Test Acc (%)": round(acc * 100, 2)
    })

df = pd.DataFrame(results)
print("\n", df.to_string(index=False))