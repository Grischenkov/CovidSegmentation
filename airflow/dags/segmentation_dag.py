import airflow

from airflow.operators.python import PythonOperator

def generate_images_lists():
    pass
def prepare_segmentation_datasets():
    pass
def load_segmentation_datasets():
    pass
def generate_segmentation_datasets():
    pass
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
    GenerateClassificationDatasets = PythonOperator(
        task_id='generate_datasets',
        python_callable=generate_segmentation_datasets,
    )
    TrainClassificationModel = PythonOperator (
        task_id='train_model',
        python_callable=train_segmentation_model,
    )
    GenerateImagesLists >> PrepareClassificationDatasets >> LoadClassificationDatasets >> GenerateClassificationDatasets >> TrainClassificationModel