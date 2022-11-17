import os

class Files:
    @staticmethod
    def get_files_list(path):
        result = []
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                result.append(file)
        return result