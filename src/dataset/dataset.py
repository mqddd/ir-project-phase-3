import os
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self):
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.news = []
        self.labels = []
        self.cat_label = dict()
        self.label_cat = dict()
        self.dataset_path = '/home/meqdad/uni/ir-project-phase3/datasets'

    def load_dataset_from_file(self, path):
        '''
        :param file_path:
        :return:
        '''
        categories = os.listdir(path)
        for i, cat in enumerate(categories):
            self.cat_label[cat] = i
            self.label_cat[i] = cat
            cat_path = path + '/' + cat
            for file in os.listdir(cat_path):
                file_path = cat_path + '/' + file
                f = open(file_path, 'rb')
                content = f.read().decode(errors='replace')
                self.news.append(content)
                self.labels.append(i)

        df = pd.DataFrame({'News': self.news, 'Labels': self.labels})
        df.to_csv(self.dataset_path + '/all_data.csv')
                
        return self

    def split_train_dev_test(self, train_per=70, dev=15, test=15):
        '''

        :param train_per: percentage of how many of samples should be in train set
        :param dev: percentage of how many of samples should be in dev set
        :param test: percentage of how many of samples should be in dev set
        :return:
        '''

        all_data = pd.read_csv(self.dataset_path + '/all_data.csv')
        x, x_test, y, y_test = train_test_split(all_data['News'], all_data['Labels'], 
            test_size=0.15, train_size=0.85)
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.18, train_size=0.82)
        train_df = pd.DataFrame({'News': x_train, 'Labels': y_train})
        train_df.to_csv(self.dataset_path + '/train_data.csv')
        self.train_set = train_df
        dev_df = pd.DataFrame({'News': x_dev, 'Labels': x_dev})
        dev_df.to_csv(self.dataset_path + '/dev_data.csv')
        self.dev_set = dev_df
        test_df = pd.DataFrame({'News': x_test, 'Labels': x_test})
        test_df.to_csv(self.dataset_path + '/test_data.csv')
        self.test_set = test_df


DATA_PATH = '/home/meqdad/uni/ir-project-phase3/data'

def prepare_dataset(config):
    ds_cfg = config.get('dataset')
    dataset = Dataset(ds_cfg).load_dataset_from_file()
    dataset.split_train_dev_test()
    return

def test():
    dataset = Dataset()
    #dataset.load_dataset_from_file(DATA_PATH)
    dataset.split_train_dev_test()
test()