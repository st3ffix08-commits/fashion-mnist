# FashionMNIST Classification

Обучение и сравнение сверточных моделей (ResNet-18, VGG-11, AlexNet) на датасете FashionMNIST.

## Установка

```bash
pip install -r requirements.txt
Структура проекта
Plaintext
├── configs/
│   └── config.yaml          # Параметры обучения и выбор модели
├── model_checkpoints/       # Сохраненные веса (.pth)
├── scripts/
│   ├── train.py             # Обучение модели
│   └── evaluate_all.py      # Сводная таблица метрик по всем чекпоинтам
├── src/
│   ├── data/                # Загрузка и предобработка FashionMNIST
│   ├── models/              # Архитектуры и ModelFactory
│   └── utils/               # Логика обучения (Trainer)
└── requirements.txt         # Зависимости проекта
Обучение
В configs/config.yaml укажи нужную архитектуру в поле model.name (resnet18, vgg11 или alexnet).

Запусти тренировку:

Bash
python -m scripts.train
Чекпоинт автоматически сохранится в папку model_checkpoints/.

Оценка и сравнение
Запуск тестирования по всем сохраненным чекпоинтам и вывод итоговой таблицы метрик:

Bash
python -m scripts.evaluate_all