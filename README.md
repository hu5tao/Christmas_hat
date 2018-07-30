# Christmas_hat
腾讯微信给头像戴帽子算法-复杂密集场景人物带帽
原理：极小人脸检测+帽子（参考：https://github.com/peiyunh/tiny） 将人脸检测的位置信息，保存在本地，然后读取位置信息进行帽子佩戴。

说明：
  环境:MATLAB MatConvNet
  
  首先将我的仓库代码保存到本地，如果遇到什么问题，也可以对 https://github.com/peiyunh/tiny 参考进行配置。

步骤：

1、下载仓库 git clone --recursive https://github.com/hu5tao/Christmas_hat.git

2、 Installing - MatConvNet 
  >>cd matconvnet/
  
  >>addpath matlab/

  >>vl_compilenn('enableImreadJpeg', true, 'enableGpu', true, 'cudaRoot', [cuda_dir],...
            'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', [cudnn_dir]);
            
  这里的cuda_dir和cudnn_dir为你要填的路径;我的是:
  >>vl_compilenn('enableImreadJpeg', true, 'enableGpu', true, 'cudaRoot', ['/usr/local/cuda-8.0'],...
            'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', ['/usr/local/cuda-8.0']);
						
  发现还是出错如下:
    错误使用 vl_compilenn>nvcc_compile (line 536) Command "/usr/local/cuda-8.0/bin/nvcc" -c "/home/hutao/huta/tiny/matconvnet/matlab  /src/bits/impl/nnconv_cudnn.cu" -DNDEBUG -DENABLE_GPU
  分析:应该是cuda的问题,或许是版本问题;但是仔细打开vl_compilenn.m文件,发现如果不指定cuda的路径(有可能CUDA路径出错),MATLAB也会有默认的CUDA路径;所以我不需要指定cuda的路径;
  
  >>vl_compilenn('enableImreadJpeg', true, 'enableGpu', true, ...
            'cudaMethod', 'nvcc', 'enableCudnn', true);
  忽视警告;直至编译成功;

  >> vl_testnn('gpu', true);  % vl_testnn('gpu', false) for cpu-only,漫长等待完成.
  
3、编译Compile our MEX function in MATLAB and test if it works as expected:

  >>cd utils/;  (而不是matconvnet/utils)
  
  >> compile_mex;
  
  >> test_compute_dense_overlap;
  
4、下载数据Download WIDER FACE and unzip data and annotation files to data/widerface such that:

  $ ls data/widerface
  
  wider_face_test.mat   
  wider_face_train.mat    
  wider_face_val.mat
  WIDER_test/          
  WIDER_train/            
  WIDER_val/
  
5、Download pretrained model,下载预训练模型:

  ResNet101
  
  ResNet50
  
  VGG16
  
  在tiny文件夹下新建一个文件夹:trained_models,将下载好的预训练模型放在复制到trained_models文件夹下;
  
**测试的Demo**：可以下载我已经训练好的模型进行测试：百度云盘：https://pan.baidu.com/s/12UZuo6H2TYK4JV6N_U_55Q 新建data文件夹，将测试数据放入
  
  bboxes = tiny_face_detector('data/demo/selfie.jpg', './selfie.png', 0.5, 0.1, 0)

将要戴帽子的图片放到主文件夹下进行检测,将检测的结果bboxes变量存为detect_results.mat文件,在matlab 命令窗口输入：

  >>save detect_results bboxes
  
最后运行：**python add_hat.py**,戴帽子结果将保存到当前文件夹中output.jpg

如果整个有问题或者需要训练的话，可以参考 tiny_face_detect_push.ipynb，结果如下：
![Image text](https://github.com/hu5tao/Christmas_hat/blob/master/output.jpg)
