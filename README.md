# Covid Segmentation

Данные для работы получены из открытых датасетов:
* [Раз](https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset)
* [Два](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

## Описание
### *Предобработка данных*

1. Изменение размера изображения до (512 х 512)
2. Преобразование цветного изображения в монохромное
3. Обработка изображения
    + Для классификации: аугментация данных (повороты изображения)
    + Для сегментации: применение выравния гистограммы (OpenCV CLAHE)


### *Сегментация*
Используется для выделения пораженных COVID-19 областей на КТ-снимках легких.

+ Архитектура:

![Сегментация](report/images/U-Net.jpg?raw=true "Сегментация")

+ Результаты:

Для оценки качества модели используется среднее значение метрики Dice на тестовом наборе данных. 

dice: 0.73

![Результат сегментации](report/images/U-Net_Sample.png?raw=true "Результат сегментации")

### *Классификация*
Используется для валидации входных данных пользователя, предупреждая о том, что пользоватеь пытается проанализировать не КТ-снимок легких или он являетсянекачественным.

+ Прогногнозируемые классы:

 | Класс | Описание                        |
 | ----- | --------                        | 
 | 0     | Не КТ-снимок легких             | 
 | 1     | Подходящий КТ-снимок легких     | 
 | 2     | Некачественный КТ-снимок легких | 

+ Архитектура:

![Классификация](report/images/ResNet.jpg?raw=true "Классификация")

+ Результаты:

 | Класс | precision | recall | f1-score |
 | ----- | -----     | ------ | -------- | 
 | 0     | 0.92      | 0.99   | 0.95     |
 | 1     | 0.98      | 0.93   | 0.96     |
 | 2     | 1         | 0.98   | 0.99     | 
accuracy: 0.97 

![Матрица ошибок](report/images/ResNet_Confusion_Matrix.png?raw=true "Матрица ошибок")

## Запуск
+ Скачать [набор данных](https://drive.google.com/file/d/1cbktnXSAqGj2VlQnN1Jlxv344WqYEA3X/view?usp=sharing)
+ Распаковать архив в корневой папке проекта
+ Добаыить файл *parameters.yaml* в /data:
```yaml
    classification:
        experiment_name: *YourExperimentName*
        model_name: *YourModelName*
        use_augmentation: False
        n_samples: 1
        test_size: 0.3
        batch: 1
        epochs: 5
        shuffle: True
        learning_rate: 0.001
    segmentation:
        experiment_name: *YourExperimentName*
        model_name: *YourModelName*
        use_clahe: True
        n_samples: 10
        test_size: 0.3
        batch: 1
        epochs: 5
        shuffle: True
        learning_rate: 0.001
```
+ Выполнить: 
```
    docker-compose up
```

## Использование
+ Запуск пайплайнов: [Airflow](http://localhost:8080/)
+ Версионирование экспериментов [MlFlow](http://localhost:5000/)