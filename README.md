# contrast_seg_paddle  
A reproduction of Exploring Cross-Image Pixel Contrast for Semantic Segmentation in PaddlePaddle    
1、本repo使用paddlepaddle复现论文：Exploring Cross-Image Pixel Contrast for Semantic Segmentation    
2、复现指标 cityscapes miou 80.18%， 40k复现的miou为81.52%，60k复现的miou为82.47%       
3、训练流程：configs目录下HRNet_W48_cityscapes_1024x512_40k.yml训练40k 最优miou 81.52%，configs目录下HRNet_W48_cityscapes_1024x512_60k.yml训练60k，最优miou 82.47%  
4、训练环境：v100*4， paddlepaddle=2.1.2     
5、40k及60k训练vdl文件在output目录下   
6、40k best_model百度云连接：链接：https://pan.baidu.com/s/1zdCFxPpEwbXe9aY7m3DRgQ 提取码：jlo6  
7、60k best_model百度云连接：链接：https://pan.baidu.com/s/13zYV83i-BjYhW4H8OmgAJg 提取码：b9ky  
8、本repo基于paddleseg实现，使用方法与paddleseg相同   
9、60k验证及训练aistudio地址：https://aistudio.baidu.com/aistudio/clusterprojectdetail/2333277  
10、由于验收标准由miou 80.18%变为82.2%， 40k训练未达到新标准，请使用60k配置进行训练    
