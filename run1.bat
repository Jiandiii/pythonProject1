@echo off
setlocal enabledelayedexpansion

:: 设置要运行的Python脚本路径
set SCRIPT=gnn_training.py

:: 定义参数列表
set ARG_GROUP_1= --dataset Cora --noise_type pair_noise  --graph_noise 0.0 --label_noise 0.0 --feature_noise 0.0
set ARG_GROUP_2= --dataset Cora --noise_type pair_noise  --graph_noise 0.3 --label_noise 0.0 --feature_noise 0.0
set ARG_GROUP_3= --dataset Cora --noise_type pair_noise  --graph_noise 0.0 --label_noise 0.3 --feature_noise 0.0
set ARG_GROUP_4= --dataset Cora --noise_type pair_noise  --graph_noise 0.0 --label_noise 0.0 --feature_noise 0.3
set ARG_GROUP_5= --dataset Cora --noise_type pair_noise  --graph_noise 0.3 --label_noise 0.3 --feature_noise 0.3




:: 创建输出目录
mkdir output 2>nul
set n=5
:: 循环运行Python脚本，每次使用不同的参数
for /l %%i in (1,1, %n%) do (
    echo Running Group %%i
    python %SCRIPT% !ARG_GROUP_%%i! >> output_%%i.log 2>&1
)

echo 
All tasks completed!
pause