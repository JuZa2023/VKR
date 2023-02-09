# Прогнозирование конечных свойств композитов (материалов)

## Задачи
1. Провести разведочный анализ предложенных данных.
2. Провести предобработку данных.
3. Обучить нескольких моделей для прогноза модуля упругости при растяжении и прочности при растяжении.
4. Написать нейронную сеть, которая будет рекомендовать соотношение матрица-наполнитель.
5. Оценить точность модели на тренировочном и тестовом датасете. 
6. Разработать приложение с интерфейсом командной строки.


## Выполнение работы
* Весь код, относящийся к анаизу данных и обучению моделей содержится в файле `materials.ipynb`.
* Приложение для прогнозирования - `materials_predict.py`.

Для установки необходимо установить пакеты из файла `requirements.txt`:
```shell
pip3 install -r requirements.txt
```

Для запуска использовать файл `materials_predict.py`, на вход которому подать `csv`-файл с признаками: `test_x.csv`. Пример его генерации см. в `materials.ipynb`.

```
Usage:
python materials_predict.py [OPTIONS]

 Predict composite materials.                                                   
 :param path_to_weights: Path    Path to trained neural network weights.       
 :param path_to_data: Path       Path to CSV file with test data.
 :param path_to_scaler_x: Path   Saved StandardScaler for features.
 :param path_to_scaler_y: Path   Saved StandardScaler for predictions.               

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --path-to-weights         PATH  [default: net.pth]                           │
│ --path-to-data            PATH  [default: test_x.csv]                        │
│ --path-to-scaler-x        PATH  [default: x_scaler.pkl]                      │
│ --path-to-scaler-y        PATH  [default: y_scaler.pkl]                      │
│ --help                          Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Файлы с весами сети, Scaler'ами для признаков и предсказаний лежат в этом же репозитории.

Пример запуска:
`python materials_predict.py`