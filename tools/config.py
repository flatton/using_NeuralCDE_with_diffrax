from typing import List

from pydantic import BaseModel

class MNISTStrokeDatasetConfig(BaseModel):
    '''
        # [MNIST digits stroke sequence data](https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data)
        The MNIST handwritten digit images transformed into a data set for sequence learning.
        This data set contains pen stroke sequences based on the original MNIST images.
    '''
    dataset_dir: str = './dataset/sequences'
    
    train_data_input_sequence: str = dataset_dir + '/trainimg-*-inputdata.txt'
    train_data_point_sequence: str = dataset_dir + '/trainimg-*-points.txt'
    train_data_target_sequence: str = dataset_dir + '/trainimg-*-targetdata.txt'
    train_label: str = dataset_dir + '/trainlabels.txt'
    
    test_data_input_sequence: str = dataset_dir + '/testimg-*-inputdata.txt'
    test_data_point_sequence: str = dataset_dir + '/testimg-*-points.txt'
    test_data_target_sequence: str = dataset_dir + '/testimg-*-targetdata.txt'
    test_label: str = dataset_dir + '/testlabels.txt'
