# Визуальный энкодер

Используется готовый [MobileNet v2-7 из ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet),
обученный на ImageNet. Файл, ревизия и SHA-256 записаны в [source.json](source.json).
Лицензия модели из исходного репозитория сохранена в [LICENSE](LICENSE).

Для сравнения берётся слой `mobilenetv20_features_pool0_fwd`: 1280 значений
после усреднения признаков. Классификационная голова на 1000 классов не
используется. Вход нормализуется согласно документации модели.

Encoder проверяет SHA-256 при первом открытии. Один экземпляр OpenCV Net
защищён блокировкой, поскольку setInput меняет его внутреннее состояние.
