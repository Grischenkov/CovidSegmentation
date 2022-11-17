FROM apache/airflow:2.4.2

USER root

RUN sudo apt-get update
RUN sudo apt install -y libgl1-mesa-glx
RUN sudo apt-get install -y libglib2.0-0

USER airflow

RUN pip install pandas pyyaml opencv-python