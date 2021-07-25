肺结节检测, 肺部分割  
  
  
Usage：  
> overwrite the config.py
  
```
|-- data 数据集  
|-- net 神经网络  
|-- utils 工具包  
   |-- config 配置文件：修改Config类，修改其中内容  
   |-- convert 将文件转换为numpy  
   |-- message 输入一些信息，日志  
   |-- save 保存预测结果，将其保存为nii等  
   |-- SegLung 肺部分割  
   |-- validation 训练预测等处理函数类  
   |-- visualization 输出一些图片，图标   
|-- predict 预测入口  predict a patient CT sclices
|-- test 测试入口  test model perform
|-- train 训练入口  
```