import random
import pandas
import imageio
import numpy as np

def _generate_random_img_paths(df, n, path_to_pic_column):
    indicies = [random.randint(0,df.shape[0]) for i in range(n)]
    return df.loc[indicies, path_to_pic_column]

def get_n_random_img_shapes(df, n=10, path_to_pic_column='img_path'):
    img_paths = _generate_random_img_paths(df, n, path_to_pic_column)
    return [
        imageio.imread(img_path).shape for img_path in img_paths
    ]
    
def get_n_random_img_unique_vals(df, n=10, path_to_pic_column='img_path'):
    img_paths = _generate_random_img_paths(df, n, path_to_pic_column)
    return [
        np.unique(
            np.array(imageio.imread(img_path))) for img_path in img_paths
    ]
    