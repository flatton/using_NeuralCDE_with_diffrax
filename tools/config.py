from typing import List

from pydantic import BaseModel, Field, field_validator, computed_field

class MNISTStrokeDatasetConfig(BaseModel):
    """MNIST digits stroke sequence data
    **MNIST digits stroke sequence data:**
    https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data
    
    ```
        The MNIST handwritten digit images transformed into a data set for sequence learning.
        This data set contains pen stroke sequences based on the original MNIST images.
    ```
    """
    dataset_dir: str = './dataset/sequences'
    
    train_data_input_sequence: str = dataset_dir + '/trainimg-*-inputdata.txt'
    train_data_point_sequence: str = dataset_dir + '/trainimg-*-points.txt'
    train_data_target_sequence: str = dataset_dir + '/trainimg-*-targetdata.txt'
    train_label: str = dataset_dir + '/trainlabels.txt'
    
    test_data_input_sequence: str = dataset_dir + '/testimg-*-inputdata.txt'
    test_data_point_sequence: str = dataset_dir + '/testimg-*-points.txt'
    test_data_target_sequence: str = dataset_dir + '/testimg-*-targetdata.txt'
    test_label: str = dataset_dir + '/testlabels.txt'

class ExperimentConfig(BaseModel):
    # Condition of Dataset
    dataset_size_train: int = Field(default=128)
    dataset_size_test: int = Field(default=128)
    noise_ratio: float = Field(default=0.0)
    input_format: str = Field(default='point_sequence', frozen=True)
    interpolation: str = Field(default='cubic', frozen=True)

    # Condition of Model
    neural_model_name: str = Field(default='NeuralCDE')
    out_size: int = Field(default=10, frozen=True)
    hidden_size: int = Field(default=16, frozen=True)
    width_size: int = Field(default=128, frozen=True)
    depth: int = Field(default=3, frozen=True)

    # Condition of Training
    batch_size: int = Field(default=32)
    lr: float = Field(default=1e-3)
    steps: int = Field(default=2)
    seed: int = Field(default=5678, frozen=True)

    # Others
    output_dir: str = Field(default='./output', frozen=True)
    output_model_dir: str = Field(default='/model', frozen=True)
    output_model_name: str = Field(default='/test_checkpoint')
    output_memray_dir: str = Field(default='/memray', frozen=True)
    output_config_dir: str = Field(default='/configuration', frozen=True)
    output_result_dir: str = Field(default='/result', frozen=True)

    class Config:
        validate_assignment = True

    @field_validator("noise_ratio")
    @classmethod
    def validate_noise_ratio(cls, noise_ratio: float) -> float:
        if (noise_ratio < 0.) or (noise_ratio > 1.):
            raise ValueError(f'`noise_ratio` must be between 0 and 1. But now {noise_ratio}')
        return noise_ratio

    @field_validator("input_format")
    @classmethod
    def validate_input_format(cls, input_format: str) -> str:
        options = ['input_sequence', 'point_sequence']
        if not input_format in options:
            raise ValueError(f'Please select one of the following options: {options}. But now {input_format}')
        return input_format

    @field_validator("interpolation")
    @classmethod
    def validate_interpolation(cls, interpolation: str) -> str:
        options = ['cubic', 'linear']
        if not interpolation in options:
            raise ValueError(f'Please select one of the following options: {options}. But now {interpolation}')
        return interpolation

    @field_validator("neural_model_name")
    @classmethod
    def validate_input_format(cls, neural_model_name: str) -> str:
        options = ['NeuralCDE', 'RNN']
        if not neural_model_name in options:
            raise ValueError(f'Please select one of the following options: {options}. But now {neural_model_name}')
        return neural_model_name

    @computed_field
    @property
    def output_model_path(self) -> str:
        return self.output_dir + self.output_model_dir + self.output_model_name + '.eqx'

    @computed_field
    @property
    def output_memray_path(self) -> str:
        return self.output_dir + self.output_memray_dir + self.output_model_name + '.bin'

    @computed_field
    @property
    def output_config_path(self) -> str:
        return self.output_dir + self.output_config_dir + self.output_model_name + '.json'

    @computed_field
    @property
    def output_result_path(self) -> str:
        return self.output_dir + self.output_result_dir + self.output_model_name + '.json'
    