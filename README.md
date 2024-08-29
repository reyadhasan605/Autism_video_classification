# Autsim_children_action_classification
Implementation of CNN LSTM with Resnet backend, Video Vision Transformer (VIVIT), and 3D CNN for Video Classification

# Getting Started
## Prerequisites
* PyTorch
* FFmpeg, FFprobe
* Python 3


### Dataset prepration

```
mkdir data
mkdir data/video_data
```
Put your video dataset inside data/video_data
It should be in this form --

```
+ data 
    + video_data    
            - arm_flipping
            - head_banging
            + spinning 
                    - spnning0.avi
                    - spinning1.avi
                    - spinning2.avi
```

Generate Images from the Video dataset
```
./utils/generate_data.sh
```

## Train
Once you have created the dataset, start training ->
```
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes <num_classes>
```

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--resume_path <path-to-model> 
```


## Tensorboard Visualisation
```
tensorboard --logdir tf_logs

```


## References
* https://github.com/kenshohara/video-classification-3d-cnn-pytorch
* https://github.com/HHTseng/video-classification


