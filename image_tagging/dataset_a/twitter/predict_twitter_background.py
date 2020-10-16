from imageai.Prediction import ImagePrediction
import os
import pandas as pd
from PIL import Image


def clear_valid_path(path):
    r = path.split('/')[7]
    return r.replace('.jpg', '')


def get_image_tags(path_from, path_to, csv):
    """reads images and returns csv containing predictions, percentages and userid"""

    # set up model
    multiple_prediction = ImagePrediction()
    multiple_prediction.setModelTypeAsResNet()
    multiple_prediction.setModelPath(os.path.join(path_model))
    multiple_prediction.loadModel()

    # create list of profile pic filenames
    pics = os.listdir(path_from)
    all_images_array = []

    # only use actual photos
    for each_file in pics:
        if each_file.endswith(".jpg") or each_file.endswith(".png"):
            all_images_array.append(each_file)

    # create path for each pic
    path_pics = [f'{path_from + pic}' for pic in all_images_array]
    valid_paths = []

    # check for each image if its broken
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

    df = pd.DataFrame(columns=['prediction', 'percentage', 'twitterusername'])
    c = 0

    # append each prediction to df
    for dict in res:
        interdf = pd.DataFrame(
            {'prediction': dict['predictions'],
             'percentage': dict['percentage_probabilities'],
             'twitterusername': valid_ids[c]}
        )
        df = df.append(interdf)
        c += 1

    df = df.reset_index(drop=True)

    if csv:
        df.to_csv(path_to)


if __name__ == '__main__':

    dataset = 'dataset_a'
    tw_standard_pic = f'../../../../data/{dataset}/twitter/twitter_profilepics/shardproducton.jpg'
    path_from = f'../../../../data/{dataset}/twitter/twitter_profilepics/'
    path_to = 'twitter_background_prediction/'
    path_model = '../../image_pred_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    csv = False

    get_image_tags(path_from, path_to, csv)