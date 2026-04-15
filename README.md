# Kaiwu Robot Vacuum PPO Training

这个仓库共享的是腾讯 Kaiwu `robot_vacuum` 项目的源码版训练工程。

目标很简单：

- 别人克隆仓库后，按同样的目录结构准备环境
- 使用官方提供的训练平台从头开始训练
- 得到接近的训练趋势

这个仓库不提供现成权重，默认就是从头训练。

## 仓库结构

- `code/`：策略代码、特征处理、训练逻辑、算法配置
- `train/`：训练环境的 Docker Compose 配置
- `dev/`：开发环境的 Docker Compose 配置

核心训练代码入口：

- `code/agent_ppo/agent.py`
- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/workflow/train_workflow.py`

## 运行前提

- Windows
- Docker Desktop
- Docker Compose
- NVIDIA GPU，且 Docker 已开启 GPU 支持
- 可以拉取 `kaiwu-pub.tencentcloudcr.com` 的镜像
- 自己的 `license.dat`
- 自己的 Kaiwu 平台配置

## 本仓库不包含的内容

出于安全和可移植性考虑，以下内容不会上传：

- 真实的 `train/.env`
- 真实的 `dev/.env`
- `license.dat`
- 训练日志
- checkpoint / backup_model

所以别人下载后，仍然需要自己准备：

- `license.dat`
- `KAIWU_PLAYER_ID`
- `KAIWU_TASK_UUID`
- `KAIWU_PUBLIC_KEY`
- `USER_ID`
- `MONITOR_ID`
- `TRACKER_ID`

## 当前训练配置

当前环境配置文件是 `code/agent_ppo/conf/train_env_conf.toml`，主要参数如下：

- `algorithm = ppo`
- `map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- `robot_count = 4`
- `charger_count = 4`
- `max_step = 1000`
- `battery_max = 200`

