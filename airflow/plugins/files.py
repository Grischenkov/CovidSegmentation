import os
import cv2

class Files:
    @staticmethod
    def get_files_list(path):
        result = []
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                result.append(file)
        return result
    @staticmethod
    def load_images(df, additional_path=''):
        result = []
        for image_name in df['image_name'].values:
            image = cv2.imread(f"{additional_path}{image_name}", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (512,512))
            result.append(image)
        return result
    @staticmethod
    def update_images_names(df):
        for i in range(len(df)):
            df['image_name'][i] = f"{Files.__get_file_name(df['image_name'][i])}.png"
        return df
    @staticmethod
    def save_images(images, df, path):
        i = 0
        for image in images:
            cv2.imwrite(f"{path}{df['image_name'][i]}", image)
            i+=1
    @staticmethod
    def __get_file_name(path):
        return os.path.splitext(os.path.basename(path))[0]