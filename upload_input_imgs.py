"""Upload all images from local folder into mongo database"""
import os
from pymongo import MongoClient
from tqdm import tqdm
from utils_data import get_secret

if __name__ == '__main__':

    INPUT_DIR = './input_imgs/'
    img_file_names = sorted(os.listdir(INPUT_DIR))

    mongo_connection_string = get_secret('mongo_connection_string')
    mongo = MongoClient(mongo_connection_string)

    for img_file_name in tqdm(img_file_names, unit='img', desc='Upload images to mongodb'):
        with open(f'{INPUT_DIR}{img_file_name}', "rb") as img_file:
            raw_img = img_file.read()
        mongo.hiro.input_imgs.insert_one({'_id': img_file_name, 'content': raw_img})
