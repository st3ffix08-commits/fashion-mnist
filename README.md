# FashionMNIST Classification

Обучение и сравнение сверточных моделей (ResNet-18, VGG-11, AlexNet) на датасете FashionMNIST.

## Установка и Запуск

```bash
pip install -r requirements.txt
В configs/config.yaml укажи нужную архитектуру в поле model.name (resnet18, vgg11 или alexnet). По умолчанию resnet18

Запусти тренировку:

Bash
python -m scripts.train
Чекпоинт автоматически сохранится в папку model_checkpoints/.

Оценка и сравнение
Запуск тестирования по всем сохраненным чекпоинтам и вывод итоговой таблицы метрик:

Bash
python -m scripts.evaluate_all

Как запустить работу и проверить свое изображение (ВАЖНО!! Модель обучена на данных FashionMNIST, так что определять может только 10 вещей)
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"

закидываешь картинку в папку images

запускаешь через консоль 

python scripts/infer.py images/test.jpg ##там лежит уже 1 тестовая картинка с интернета