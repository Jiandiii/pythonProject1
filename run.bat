@echo off
setlocal enabledelayedexpansion

:: 设置要运行的Python脚本路径
set SCRIPT=main_ali.py

:: 定义参数列表
set ARG_GROUP_1= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.0 --label_noise 0.0 --feature_noise 0.0 --teacher_student teacher --K 200 --num_parts 4
set ARG_GROUP_2= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.0 --label_noise 0.0 --feature_noise 0.0 --teacher_student student --K 200 --num_parts 4
set ARG_GROUP_3= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.1 --label_noise 0.1 --feature_noise 0.1 --teacher_student teacher --K 200 --num_parts 4
set ARG_GROUP_4= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.1 --label_noise 0.1 --feature_noise 0.1 --teacher_student student --K 200 --num_parts 4
set ARG_GROUP_5= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.3 --label_noise 0.3 --feature_noise 0.3 --teacher_student teacher --K 200 --num_parts 4
set ARG_GROUP_6= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.3 --label_noise 0.3 --feature_noise 0.3 --teacher_student student --K 200 --num_parts 4
set ARG_GROUP_7= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.0 --label_noise 0.0 --feature_noise 0.0 --teacher_student teacher --K 200 --num_parts 4
set ARG_GROUP_8= --dataset ogbn-arxiv --noise_type uniform  --graph_noise 0.2 --label_noise 0.2 --feature_noise 0.2 --teacher_student teacher --K 200 --num_parts 4

:: 创建输出目录
mkdir output 2>nul
@REM set n3=24
@REM :: 循环运行Python脚本，每次使用不同的参数
@REM for /l %%i in (1,1, %n3%) do (
@REM     echo Running Group %%i
@REM     python %SCRIPT% !ARG_GROUP_%%i! >> output_%%i.log 2>&1
@REM )

@REM set n1=62
@REM :: 循环运行Python脚本，每次使用不同的参数
@REM for /l %%i in (62,1, %n1%) do (
@REM     echo Running Group %%i
@REM     python %SCRIPT% !ARG_GROUP_%%i! >> output_%%i.log 2>&1
@REM )

@REM set n2=68
@REM :: 循环运行Python脚本，每次使用不同的参数
@REM for /l %%i in (67,1, %n2%) do (
@REM     echo Running Group %%i
@REM     python %SCRIPT% !ARG_GROUP_%%i! >> output_%%i.log 2>&1
@REM )

set n4=8
:: 循环运行Python脚本，每次使用不同的参数
for /l %%i in (6,1, %n4%) do (
    echo Running Group %%i
    python %SCRIPT% !ARG_GROUP_%%i! >> output_%%i.log 2>&1
)
echo 
All tasks completed!
pause