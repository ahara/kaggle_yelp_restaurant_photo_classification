from os.path import join


# Raw input data
DATA_PATH = '/home/adam/Projects/Yelp_data/data'

P2B_TRAIN = join(DATA_PATH, 'train_photo_to_biz_ids.csv')
LABELS_TRAIN = join(DATA_PATH, 'train.csv')
PHOTOS_TRAIN = join(DATA_PATH, 'train_photos')

P2B_TEST = join(DATA_PATH, 'test_photo_to_biz.csv')
PHOTOS_TEST = join(DATA_PATH, 'test_photos')

# Storage
STORAGE_PATH = '/home/adam/Projects/Yelp/storage'
#STORAGE_DATA_PATH = join(STORAGE_PATH, 'data')
STORAGE_DATA_PATH = join(STORAGE_PATH, 'models')
STORAGE_MODELS_PATH = join(STORAGE_PATH, 'models')

# Output
OUTPUT_PATH = '/home/adam/Projects/Yelp/output'
SUBMISSION_PATH = join(OUTPUT_PATH, 'submission.csv')