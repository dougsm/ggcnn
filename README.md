**Note:** This is a cleaned-up, PyTorch port of the GG-CNN code.  For the original Keras implementation, see the `RSS2018` branch.  
Main changes are major code clean-ups and documentation, an improved GG-CNN2 model, ability to use the Jacquard dataset and simpler evaluation.    


# Generative Grasping CNN (GG-CNN)

The GG-CNN is a lightweight, fully-convolutional network which predicts the quality and pose of antipodal grasps at every pixel in an input depth image.  The lightweight and single-pass generative nature of GG-CNN allows for fast execution and closed-loop control, enabling accurate grasping in dynamic environments where objects are moved during the grasp attempt.

This repository contains the implementation of the Generative Grasping Convolutional Neural Network (GG-CNN) from the paper:

**Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach**

*[Douglas Morrison](http://dougsm.com), [Peter Corke](http://petercorke.com), [Jürgen Leitner](http://juxi.net)*

Robotics: Science and Systems (RSS) 2018

[arXiv](https://arxiv.org/abs/1804.05172) | [Video](https://www.youtube.com/watch?v=7nOoxuGEcxA)

If you use this work, please cite:

```text
@inproceedings{morrison2018closing,
	title={{Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach}},
	author={Morrison, Douglas and Corke, Peter and Leitner, J\"urgen},
	booktitle={Proc.\ of Robotics: Science and Systems (RSS)},
	year={2018}
}
```

**Contact**

Any questions or comments contact [Doug Morrison](mailto:doug.morrison@roboticvision.org).

## Installation

This code was developed with Python 3.6 on Ubuntu 16.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets

Currently, both the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/) are supported.

### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

## Pre-trained Models

**Coming Soon**

## Training

Training is done by the `train_ggcnn.py` script.  Run `train_ggcnn.py --help` to see a full list of options, such as dataset augmentation and validation options.

Some basic examples:

```bash
# Train GG-CNN on Cornell Dataset
python train_ggcnn.py --description training_example --network ggcnn --dataset cornell --dataset-path <Path To Dataset>

# Train GG-CNN2 on Jacquard Datset
python train_ggcnn.py --description training_example2 --network ggcnn2 --dataset jacquard --dataset-path <Path To Dataset>
```

Trained models are saved in `output/models` by default, with the validation score appended.

## Evaluation/Visualisation

Evaluation or visualisation of the trained networks are done using the `eval_ggcnn.py` script.  Run `eval_ggcnn.py --help` for a full set of options.

Important flags are:
* `--iou-eval` to evaluate using the IoU between grasping rectangles metric.
* `--jacquard-output` to generate output files in the format required for simulated testing against the Jacquard dataset.
* `--vis` to plot the network output and predicted grasping rectangles.

For example:

```bash
python eval_ggcnn.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --jacquard-output --iou-eval
```


## Running on a Robot

Our ROS implementation for running the grasping system see [https://github.com/dougsm/mvp_grasp](https://github.com/dougsm/mvp_grasp).

The original implementation for running experiments on a Kinva Mico arm can be found in the repository [https://github.com/dougsm/ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping).
