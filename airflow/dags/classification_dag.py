import os
import yaml
import airflow
import pandas as pd

from files import Files
from dataset import Dataset
from airflow.operators.python import PythonOperator

def generate_images_lists():
    pd.DataFrame(data=Files.get_files_list('data/source/CT/'), columns=['image_name']).to_csv('data/source/CT.csv', index=False)
    pd.DataFrame(data=Files.get_files_list('data/source/NonCT/'), columns=['image_name']).to_csv('data/source/NonCT.csv', index=False)
def prepare_classification_datasets():
    parameters = yaml.safe_load(open('data/parameters.yaml'))
    ct = Dataset.shuffle_list('data/source/CT.csv')
    non_ct = Dataset.shuffle_list('data/source/NonCT.csv')
    data = []
    for i in range(parameters['classification']['n_samples']):
        if i % 2 == 0:
            data.append([os.path.join('data/source/NonCT', non_ct.pop()), 0])
        else:
            data.append([os.path.join('data/source/CT', ct.pop()), 1])
        if len(non_ct) == 0 or len(ct) == 0:
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
    pass
def generate_classification_datasets():
    pass
def train_classification_model():
    pass

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
    GenerateImagesLists >> PrepareClassificationDatasets >> LoadClassificationDatasets >> AugmentClassificationDatasets >> GenerateClassificationDatasets >> TrainClassificationModel