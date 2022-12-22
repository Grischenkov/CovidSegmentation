import os
import cv2
import yaml
import random
import mlflow
import airflow
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from files import Files
from dataset import Dataset
import segmentation_model

from airflow.operators.python import PythonOperator

def generate_images_lists():
    pd.DataFrame(data=Files.get_files_list('data/source/CT/'), columns=['image_name']).to_csv('data/source/CT.csv', index=False)
    pd.DataFrame(data=Files.get_files_list('data/source/NonCT/'), columns=['image_name']).to_csv('data/source/NonCT.csv', index=False)
def prepare_segmentation_datasets():
    parameters = yaml.safe_load(open('data/parameters.yaml'))
    ct = Dataset.shuffle_list('data/source/CT.csv')
    ct_images = []
    masks_images = []
    for i in range(parameters['segmentation']['n_samples']):
        name = ct.pop()
        ct_images.append([os.path.join('data/source/CT', name)])
        masks_images.append([os.path.join('data/source/MASK', name)])
        if len(ct) == 0:
            break
    ct_df = pd.DataFrame(ct_images, columns=['image_name'])
    masks_df = pd.DataFrame(masks_images, columns=['image_name'])
    ct_train_df, ct_test_df = Dataset.split_dataset(ct_df, parameters['segmentation']['test_size'])
    masks_train_df, masks_test_df = Dataset.split_dataset(masks_df, parameters['segmentation']['test_size'])
    ct_train_df.to_csv('data/segmentation/ct_train_df.csv', index=False)
    ct_test_df.to_csv('data/segmentation/ct_test_df.csv', index=False)
    masks_train_df.to_csv('data/segmentation/masks_train_df.csv', index=False)
    masks_test_df.to_csv('data/segmentation/masks_test_df.csv', index=False)
def load_segmentation_datasets():
    ct_train_df = pd.read_csv('data/segmentation/ct_train_df.csv')
    ct_test_df = pd.read_csv('data/segmentation/ct_test_df.csv')
    masks_train_df = pd.read_csv('data/segmentation/masks_train_df.csv')
    masks_test_df = pd.read_csv('data/segmentation/masks_test_df.csv')
    train_ct_images = Files.load_images(ct_train_df)
    train_masks_images = Files.load_images(masks_train_df)
    test_ct_images = Files.load_images(ct_test_df)
    test_masks_images = Files.load_images(masks_test_df)
    ct_train_df = Files.update_images_names(ct_train_df)
    ct_test_df = Files.update_images_names(ct_test_df)
    masks_train_df = Files.update_images_names(masks_train_df)
    masks_test_df = Files.update_images_names(masks_test_df)
    ct_train_df.to_csv('data/segmentation/ct_train_df.csv', index=False)
    ct_test_df.to_csv('data/segmentation/ct_test_df.csv', index=False)
    masks_train_df.to_csv('data/segmentation/masks_train_df.csv', index=False)
    masks_test_df.to_csv('data/segmentation/masks_test_df.csv', index=False)
    Files.save_images(train_ct_images, ct_train_df, 'data/segmentation/train/CT/')
    Files.save_images(test_ct_images, ct_test_df, 'data/segmentation/test/CT/')
    Files.save_images(train_masks_images, masks_train_df, 'data/segmentation/train/MASK/')
    Files.save_images(test_masks_images, masks_test_df, 'data/segmentation/test/MASK/')
def clahe_segmentation_datasets():
    params = yaml.safe_load(open('data/parameters.yaml'))
    if params['segmentation']['use_clahe'] == False:
        return
    ct_train_df = pd.read_csv('data/segmentation/ct_train_df.csv')
    ct_test_df = pd.read_csv('data/segmentation/ct_test_df.csv')
    train_ct_images = np.array(Files.load_images(ct_train_df, 'data/segmentation/train/CT/'))
    test_ct_images = np.array(Files.load_images(ct_test_df, 'data/segmentation/test/CT/'))
    clahe = cv2.createCLAHE(clipLimit=7.0)
    for i in range(len(train_ct_images)):
        train_ct_images[i] = clahe.apply(train_ct_images[i])
    for i in range(len(test_ct_images)):
        test_ct_images[i] = clahe.apply(test_ct_images[i])
    Files.save_images(train_ct_images, ct_train_df, 'data/segmentation/train/CT/')
    Files.save_images(test_ct_images, ct_test_df, 'data/segmentation/test/CT/')
def generate_segmentation_datasets():
    X_train = np.array(Files.load_images(pd.read_csv('data/segmentation/ct_train_df.csv'), 'data/segmentation/train/CT/')) / 255
    Y_train = np.array(Files.load_images(pd.read_csv('data/segmentation/masks_train_df.csv'), 'data/segmentation/train/MASK/')) / 255
    Files.save_npy('data/segmentation/train/X_train.npy', X_train)
    Files.save_npy('data/segmentation/train/Y_train.npy', Y_train)
    X_test = np.array(Files.load_images(pd.read_csv('data/segmentation/ct_test_df.csv'), 'data/segmentation/test/CT/')) / 255
    Y_test = np.array(Files.load_images(pd.read_csv('data/segmentation/masks_test_df.csv'), 'data/segmentation/test/MASK/')) / 255
    Files.save_npy('data/segmentation/test/X_test.npy', X_test)
    Files.save_npy('data/segmentation/test/Y_test.npy', Y_test)
def train_segmentation_model():
    parameters = yaml.safe_load(open('data/parameters.yaml'))
    mlflow.set_tracking_uri('http://host.docker.internal:5000')
    mlflow.set_experiment('segmentation')

    X_train = Files.load_npy('data/segmentation/train/X_train.npy')
    Y_train = Files.load_npy('data/segmentation/train/Y_train.npy')
    X_test = Files.load_npy('data/segmentation/test/X_test.npy')
    Y_test = Files.load_npy('data/segmentation/test/Y_test.npy')

    model = segmentation_model.model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['segmentation']['learning_rate']), 
            loss=segmentation_model.dice_loss, 
            metrics=[segmentation_model.dice])

    with mlflow.start_run(run_name=parameters['segmentation']['experiment_name']):
        mlflow.log_param('CLAHE', parameters['segmentation']['use_clahe'])
        mlflow.log_param('n_samples', parameters['segmentation']['n_samples'])
        mlflow.log_param('test_size', parameters['segmentation']['test_size'])
        mlflow.log_param('epochs', parameters['segmentation']['epochs'])
        mlflow.log_param('batch', parameters['segmentation']['batch'])
        mlflow.log_param('shuffle', parameters['segmentation']['shuffle'])
        mlflow.log_param('optimizer', model.optimizer.name)
        mlflow.log_param('learning_rate', parameters['segmentation']['learning_rate'])
        
        history = model.fit(X_train, Y_train, epochs=parameters['segmentation']['epochs'], shuffle=parameters['segmentation']['shuffle'])
        
        pred = list()
        true = Y_test.tolist()
        avg_dice = 0
        for i in range(len(X_test)):
            pred.append(model.predict(X_test[i].reshape(1, 512, 512, 1)).reshape(512, 512))
            avg_dice += segmentation_model.dice(np.array(true[i]).astype('float32'), np.array(pred[i]).astype('float32'))
        avg_dice /= len(X_test)
        avg_dice = avg_dice.numpy()
        mlflow.log_metric('avg test dice', avg_dice)
        
        for i in range(parameters['segmentation']['epochs']):
            mlflow.log_metric('loss', history.history['loss'][i], step=i)
            mlflow.log_metric('dice', history.history['dice'][i], step=i)
        
        with open('model_summary.txt', 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))
        mlflow.log_artifact('model_summary.txt')        
        
        indices = random.choices(range(len(X_test)), k=5)
        fig, axes = plt.subplots(3, 5, figsize=(15,9))

        for ii, idx in enumerate(indices) :
            axes[0][ii].imshow(X_test[idx], cmap='bone')
            axes[0][ii].set_title('CT image'); plt.grid(None)
            axes[0][ii].set_xticks([]); axes[0][ii].set_yticks([])
            
            axes[1][ii].imshow(true[idx], cmap='gray')
            axes[1][ii].set_title('COVID mask'); plt.grid(None)
            axes[1][ii].set_xticks([]); axes[1][ii].set_yticks([])

            axes[2][ii].imshow(pred[idx], cmap='gray')
            axes[2][ii].set_title('Predicted COVID mask'); plt.grid(None)
            axes[2][ii].set_xticks([]); axes[2][ii].set_yticks([])
        fig.savefig('example_prediction.png')
        mlflow.log_artifact('example_prediction.png')

        mlflow.log_artifact('./plugins/segmentation_model.py')
        
        model.save('segmenter.h5')
        mlflow.log_artifact('segmenter.h5')

        mlflow.keras.log_model(keras_model=model, artifact_path='segmentation_models')
        mlflow.end_run()
def clear_segmentation_files():
    os.remove('data/segmentation/ct_train_df.csv')
    os.remove('data/segmentation/ct_test_df.csv')
    os.remove('data/segmentation/masks_train_df.csv')
    os.remove('data/segmentation/masks_test_df.csv')
    os.remove('data/segmentation/train/X_train.npy')
    os.remove('data/segmentation/train/Y_train.npy')
    os.remove('data/segmentation/test/X_test.npy')
    os.remove('data/segmentation/test/Y_test.npy')
    for path in os.listdir('data/segmentation/test/CT/'):
        os.remove(f'data/segmentation/test/CT/{path}')
    for path in os.listdir('data/segmentation/train/CT/'):
        os.remove(f'data/segmentation/train/CT/{path}')
    for path in os.listdir('data/segmentation/test/MASK/'):
        os.remove(f'data/segmentation/test/MASK/{path}')
    for path in os.listdir('data/segmentation/train/MASK/'):
        os.remove(f'data/segmentation/train/MASK/{path}')

with airflow.DAG(
    'segmentation',
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
    PrepareSegmentationDatasets = PythonOperator(
        task_id='prepare_datasets',
        python_callable=prepare_segmentation_datasets,
    )
    LoadSegmentationDatasets = PythonOperator(
        task_id='load_datasets',
        python_callable=load_segmentation_datasets,
    )
    ClaheSegmentationDatasets = PythonOperator(
        task_id='clahe_dataset',
        python_callable=clahe_segmentation_datasets,
    )
    GenerateSegmentationDatasets = PythonOperator(
        task_id='generate_datasets',
        python_callable=generate_segmentation_datasets,
    )
    TrainSegmentationModel = PythonOperator (
        task_id='train_model',
        python_callable=train_segmentation_model,
    )
    ClearRunFiles = PythonOperator (
        task_id='clear_files',
        python_callable=clear_segmentation_files,
    )
    GenerateImagesLists >> PrepareSegmentationDatasets >> LoadSegmentationDatasets >> ClaheSegmentationDatasets >> GenerateSegmentationDatasets >> TrainSegmentationModel >> ClearRunFiles