import os
import cv2
import yaml
import mlflow
import airflow
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from files import Files
from dataset import Dataset
import classification_model

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from airflow.operators.python import PythonOperator

def generate_images_lists():
    pd.DataFrame(data=Files.get_files_list('data/source/CT/'), columns=['image_name']).to_csv('data/source/CT.csv', index=False)
    pd.DataFrame(data=Files.get_files_list('data/source/NonCT/'), columns=['image_name']).to_csv('data/source/NonCT.csv', index=False)
    pd.DataFrame(data=Files.get_files_list('data/source/TrashCT/'), columns=['image_name']).to_csv('data/source/TrashCT.csv', index=False)
def prepare_classification_datasets():
    parameters = yaml.safe_load(open('data/parameters.yaml'))
    ct = Dataset.shuffle_list('data/source/CT.csv')
    non_ct = Dataset.shuffle_list('data/source/NonCT.csv')
    trash_ct = Dataset.shuffle_list('data/source/TrashCT.csv')
    data = []
    j = 0
    for i in range(parameters['classification']['n_samples']):
        j += 3
        data.append([os.path.join('data/source/NonCT', non_ct.pop()), 0])
        data.append([os.path.join('data/source/CT', ct.pop()), 1])
        data.append([os.path.join('data/source/TrashCT', trash_ct.pop()), 2])
        if len(trash_ct) == 0 or len(non_ct) == 0 or len(ct) == 0 or (j+3) >= parameters['classification']['n_samples']:
            break
    df = pd.DataFrame(data, columns=['image_name', 'class'])
    train_df, test_df = Dataset.split_dataset(df, parameters['classification']['test_size'])
    train_df.to_csv('data/classification/train_df.csv', index=False)
    test_df.to_csv('data/classification/test_df.csv', index=False)
def load_classification_datasets():
    train_df = pd.read_csv('data/classification/train_df.csv')
    test_df = pd.read_csv('data/classification/test_df.csv')
    train_images = Files.load_images(train_df)
    test_images = Files.load_images(test_df)
    train_df = Files.update_images_names(train_df)
    test_df = Files.update_images_names(test_df)
    train_df.to_csv('data/classification/train_df.csv', index=False)
    test_df.to_csv('data/classification/test_df.csv', index=False)
    Files.save_images(train_images, train_df, 'data/classification/train/')
    Files.save_images(test_images, test_df, 'data/classification/test/')
def augment_classification_datasets():
    params = yaml.safe_load(open('data/parameters.yaml'))
    if params['classification']['use_augmentation'] == False:
        return
    train_df = pd.read_csv('data/classification/train_df.csv')
    train_images = Files.load_images(train_df, 'data/classification/train/')
    size = len(train_images)
    for i in range(size):
        (h, w) = train_images[i].shape[:2]
        for j in range(1, 4):
            train_images.append(cv2.warpAffine(train_images[i], cv2.getRotationMatrix2D((w / 2, h / 2), 90 * j, 1.0), (w, h)))
            train_df = train_df.append({'image_name':f"{Files.get_file_name(train_df['image_name'][i])}_{90 * j}.png", 'class':train_df['class'][i]}, ignore_index=True)
    train_df.to_csv('data/classification/train_df.csv', index=False)
    Files.save_images(train_images, train_df, 'data/classification/train/')
def generate_classification_datasets():
    X_train, Y_train = Files.load_images_with_labels('data/classification/train/', pd.read_csv('data/classification/train_df.csv'))
    Files.save_npy('data/classification/train/X_train.npy', X_train)
    Files.save_npy('data/classification/train/Y_train.npy', Y_train)
    X_test, Y_test = Files.load_images_with_labels('data/classification/test/', pd.read_csv('data/classification/test_df.csv'))
    Files.save_npy('data/classification/test/X_test.npy', X_test)
    Files.save_npy('data/classification/test/Y_test.npy', Y_test)   
def train_classification_model():
    parameters = yaml.safe_load(open('data/parameters.yaml'))
    mlflow.set_tracking_uri('http://host.docker.internal:5000')
    mlflow.set_experiment('classification')

    X_train = Files.load_npy('data/classification/train/X_train.npy')
    Y_train = Files.load_npy('data/classification/train/Y_train.npy')
    X_test = Files.load_npy('data/classification/test/X_test.npy')
    Y_test = Files.load_npy('data/classification/test/Y_test.npy')

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(parameters['classification']['batch'])
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(parameters['classification']['batch'])

    model = classification_model.model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=parameters['classification']['learning_rate']), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=['sparse_categorical_accuracy'])

    print(model.optimizer.__dict__)

    with mlflow.start_run(run_name=parameters['classification']['experiment_name']):
        mlflow.log_param('augmentation', parameters['classification']['use_augmentation'])
        mlflow.log_param('n_samples', parameters['classification']['n_samples'])
        mlflow.log_param('test_size', parameters['classification']['test_size'])
        mlflow.log_param('epochs', parameters['classification']['epochs'])
        mlflow.log_param('batch', parameters['classification']['batch'])
        mlflow.log_param('shuffle', parameters['classification']['shuffle'])
        mlflow.log_param('optimizer', model.optimizer.name)
        mlflow.log_param('learning_rate', parameters['classification']['learning_rate'])
        
        history = model.fit(train_ds, epochs=parameters['classification']['epochs'], shuffle=parameters['classification']['shuffle'])
        predict = model.predict(test_ds)
        
        pred = list()
        true = Y_test.tolist()
        for i in range(len(predict)): 
            pred.append(np.argmax(predict[i]))
        
        for i in range(parameters['classification']['epochs']):
            mlflow.log_metric('loss', history.history['loss'][i], step=i)
            mlflow.log_metric('sparse_categorical_accuracy', history.history['sparse_categorical_accuracy'][i], step=i)
        
        with open('model_summary.txt', 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))
        mlflow.log_artifact('model_summary.txt')
        
        with open('classification_report.txt', 'w') as file:
            file.write(classification_report(true, pred))
        mlflow.log_artifact('classification_report.txt')
        
        f, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(true, pred, colorbar=False, ax=ax)
        f.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        mlflow.log_artifact('./plugins/classification_model.py')

        model.save('classifier.h5')
        mlflow.log_artifact('classifier.h5')
        
        mlflow.keras.log_model(keras_model=model, artifact_path='classification_models')
        mlflow.end_run()
def clear_classification_files():
    os.remove('data/source/CT.csv')
    os.remove('data/source/NonCT.csv')
    os.remove('data/classification/test_df.csv')
    os.remove('data/classification/train_df.csv')
    for path in os.listdir('data/classification/test/'):
        os.remove(f'data/classification/test/{path}')
    for path in os.listdir('data/classification/train/'):
        os.remove(f'data/classification/train/{path}')

with airflow.DAG(
    'classification',
    default_args={
        'owner': 'nikita',
        'start_date': airflow.utils.dates.days_ago(1)
    },
    schedule=None,
    catchup=False,
) as dag:
    GenerateImagesLists = PythonOperator(
        task_id='generate_images_lists',
        python_callable=generate_images_lists,
    )
    PrepareClassificationDatasets = PythonOperator(
        task_id='prepare_datasets',
        python_callable=prepare_classification_datasets,
    )
    LoadClassificationDatasets = PythonOperator(
        task_id='load_datasets',
        python_callable=load_classification_datasets,
    )
    AugmentClassificationDatasets = PythonOperator(
        task_id='augment_dataset',
        python_callable=augment_classification_datasets,
    )
    GenerateClassificationDatasets = PythonOperator(
        task_id='generate_datasets',
        python_callable=generate_classification_datasets,
    )
    TrainClassificationModel = PythonOperator (
        task_id='train_model',
        python_callable=train_classification_model,
    )
    ClearRunFiles = PythonOperator (
        task_id='clear_files',
        python_callable=clear_classification_files,
    )
    GenerateImagesLists >> PrepareClassificationDatasets >> LoadClassificationDatasets >> AugmentClassificationDatasets >> GenerateClassificationDatasets >> TrainClassificationModel >> ClearRunFiles