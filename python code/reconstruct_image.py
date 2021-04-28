import pandas
import numpy as np


def reconstruct_image(df, fg_df, image_size):
    image_df = df.merge(fg_df, how='outer')
    image_df.fillna(0, inplace=True)
    image = image_df['Intensity'].values[:, np.newaxis]
    image = image.reshape((image_size, -1))
    return image_df, image
