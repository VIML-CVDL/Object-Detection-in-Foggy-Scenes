### Code for paper ["Object-Detection-in-Foggy-Scenes-by-Embedding-Depth-and-Reconstruction-into-Domain-Adaptation"](https://openaccess.thecvf.com/content/ACCV2022/papers/Yang_Object_Detection_in_Foggy_Scenes_by_Embedding_Depth_and_Reconstruction_ACCV_2022_paper.pdf)


### Preparation
This paper follows the environment of [da-faster-rcnn-PyTorch](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch/)

#### Requirements: Python=3.6 and Pytorch=0.4.0

1. Download dataset
   
   - [Cityscapes and Foggy-Cityscapes](https://www.cityscapes-dataset.com/downloads/)  
   - [RTTS](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)
   - [Foggy Driving](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)
   - [SeeingThroughFog](https://web.media.mit.edu/~guysatat/fog/)
   
   - Please ensure all your datasets follows VOC style
   
   
   
   
### Train and Test

1.train the model,you need to download the pretrained res101 model [Res-101](https://github.com/jwyang/faster-rcnn.pytorch) and put it under ./model/pretrained

2.change the dataset root paths in ./lib/datasets/{whichever dataset you are using. For example, cityscape.py}

3.set up the dataset paths in ./lib/roi_da_data_layer/roidb.py line 36-45

4.Train the model
 ```Shell
 # train cityscapes -> cityscapes-foggy
 CUDA_VISIBLE_DEVICES=GPU_ID python train.py --dataset cityscape --net res101 --bs 1 --lr 2e-3 --lr_decay_step 6 --cuda
 
 # Test model in target domain 
 CUDA_VISIBLE_DEVICES=GPU_ID python lib/test.py --dataset cityscape --part test_t --net res101_transmission --model_dir {your model path}
