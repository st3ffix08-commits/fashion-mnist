from pathlib import Path
import yaml
import torch
import torch.nn as nn

from src.data.dataset import get_dataloaders
from src.models.factory import ModelFactory
from src.utils.trainer import Trainer

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

set_seed(config['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_loader, test_loader = get_dataloaders(
    batch_size=config['data']['batch_size'],
    root=config['data']['root']
)

model = ModelFactory.get_model(
    config['model']['name'],
    num_classes=config['model']['num_classes'],
    in_channels=config['model']['in_channels']
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device)
trainer.fit(train_loader, test_loader, epochs=config['training']['epochs'])

save_dir = Path(config['training']['save_dir'])
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f'{config['model']['name']}_fashion.pth'

torch.save(model.state_dict(), save_path)
print(f'model saved to {save_path}')