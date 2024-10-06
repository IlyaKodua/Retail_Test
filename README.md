# Retail_Test

### Метод
В качестве метода был выбран: косинусное расстоение между эмбеддингами, получеными из предобученной модели.
### Модель
В качестве модели была выбрана - [модель RuCLIP обученная на русском языке](https://github.com/ai-forever/ru-clip "ссылка"). Так как фотографии содержат много текста на этикетках, а модель clip очень чувствительна к тексту на картинках, то было принято решение взять ее, причем обученную на русском (тк этикетки часто содержат русские слова, и я посмотрел это действительно давало чуть более лучший результат).
### Запуск скриптов
В задаче было написано оформить решение в виде класса - это в файле `model.py`.
Файлик `embeddings_save.py` запускает чтение файлов и сохранение эмбеддингов, чтобы каждый раз их не рассчитывать.
```python
python3 embeddings_save.py # Не забыть указать путь к датасетув внутри файла.
```
В ноутбуке accuracy_p_r_plot.ipynb приведен результат и рассчет accuracy, precision, recall и подбор оптимального порога для accuracy.
### Результаты
Результаты также можно просмотреть в `accuracy_p_r_plot.ipynb` 
| Порог | Accuracy | Precision | Recall |
|---|---|---|---|
| 0.75 | 0.98 | 0.99 | 0.98 |
### Дополнительно
Я также пытался обучать веса с помощью триплетов, но это не дало хорошего результат, а также я брал другие модели, например MobileNetV2 и DinoV2.


