import os
import yaml
import airflow
import numpy as np
import pandas as pd

from files import Files
from dataset import Dataset

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
    X_train = np.array(Files.load_images(ct_train_df))
    Y_train = np.array(Files.load_images(masks_train_df))
    Files.save_npy('data/segmentation/train/X_train.npy', X_train)
    Files.save_npy('data/segmentation/train/Y_train.npy', Y_train)
    X_test = np.array(Files.load_images(ct_test_df))
    Y_test = np.array(Files.load_images(masks_test_df))
    Files.save_npy('data/segmentation/test/X_test.npy', X_test)
    Files.save_npy('data/segmentation/test/Y_test.npy', Y_test)
def train_segmentation_model():
    pass

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
    PrepareClassificationDatasets = PythonOperator(
        task_id='prepare_datasets',
        python_callable=prepare_segmentation_datasets,
    )
    LoadClassificationDatasets = PythonOperator(
        task_id='load_datasets',
        python_callable=load_segmentation_datasets,
    )
    TrainClassificationModel = PythonOperator (
        task_id='train_model',
        python_callable=train_segmentation_model,
    )
    GenerateImagesLists >> PrepareClassificationDatasets >> LoadClassificationDatasets >> TrainClassificationModel