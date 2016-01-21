from os.path import join


DATA_PATH = '/home/adam/Projects/Yelp_data/data'
STORAGE_PATH = '/home/adam/Projects/Yelp/storage'

P2B_TRAIN = join(DATA_PATH, 'train_photo_to_biz_ids.csv')
LABELS_TRAIN = join(DATA_PATH, 'train.csv')

PHOTOS_TRAIN = join(DATA_PATH, 'train_photos')