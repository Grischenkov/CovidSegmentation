import airflow

from airflow.operators.python import PythonOperator

def generate_images_lists():
    pass
def prepare_classification_datasets():
    pass
def load_classification_datasets():
    pass
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