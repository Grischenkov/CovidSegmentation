import random
import pandas as pd

class Dataset:
    @staticmethod
    def split_dataset(df, test_size=0):
        test_size = int(len(df) * test_size)
        test_iamges = df[:test_size]
        train_images = df[test_size:]
        return(train_images, test_iamges)
    
    @staticmethod
    def shuffle_list(path):
        ct = pd.read_csv(path)['image_name'].to_list()
        random.shuffle(ct)
        return (ct)