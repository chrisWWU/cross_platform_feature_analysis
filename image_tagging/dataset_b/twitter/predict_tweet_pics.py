from imageai.Prediction import ImagePrediction
import os
from pathlib import Path
import pandas as pd
from PIL import Image


def clear_valid_path(path):
    r = path.split('/')[9]
    r.replace('tweets.jpg', '')
    return r.replace('.jpg', '')


def get_image_tags(path_from, path_to, csv):
    """reads images and returns csv for each user containing predictions, percentages and image id"""

    # set up model
    multiple_prediction = ImagePrediction()
    multiple_prediction.setModelTypeAsResNet()
    multiple_prediction.setModelPath(os.path.join(path_model))
    multiple_prediction.loadModel()

    # create destination folder
    Path(path_to).mkdir(parents=True, exist_ok=True)

    # create list of all foldernames (flickr user ids)
    folders = os.listdir(path_from)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')

    # iterate through folder (user)
    j = 0
    for folder in folders:
        print(f'{j+1} / {len(folders)} -> {folder}')
        j += 1

        # create list of image names (image ids)
        pics = os.listdir(path_from + folder)
        all_images_array = []

        # only use actual photos
        for each_file in pics:
            if each_file.endswith(".jpg") or each_file.endswith(".png"):
                all_images_array.append(each_file)

        # create paths for photos
        path_pics = [f'{path_from + folder}/{pic}' for pic in all_images_array]
        valid_paths = []

        # check if image is broken
        for path in path_pics:
            try:
                im = Image.open(path)
                valid_paths.append(path)

            except IOError:
                print(f'{path}: image is broken')

        # create list of valid ids from valid paths list
        valid_ids = [clear_valid_path(x) for x in valid_paths]

        # predict valid paths
        res = multiple_prediction.predictMultipleImages(valid_paths, result_count_per_image=5)

        df = pd.DataFrame(columns=['prediction', 'percentage', 'image_id'])
        c = 0

        # append each prediction to df
        for dict in res:
            interdf = pd.DataFrame(
                {'prediction': dict['predictions'],
                 'percentage': dict['percentage_probabilities'],
                 'image_id': valid_ids[c]}
            )
            df = df.append(interdf)
            c += 1

        df = df.reset_index(drop=True)

        if csv:
            df.to_csv(path_to + folder + '.csv')


if __name__ == '__main__':

    dataset = 'dataset_b'
    path_from = f'../../../../data/{dataset}/twitter/tweet_pics/'
    path_to = 'tweet_pic_tags/'
    path_model = '../../image_pred_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    csv = False

    get_image_tags(path_from, path_to, csv)