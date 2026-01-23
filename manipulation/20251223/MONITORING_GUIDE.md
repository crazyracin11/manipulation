# 训练监控指南

## 🎯 快速开始

### 1. **快速状态检查**
```bash
./quick_status.sh
```
立即查看训练进程状态、GPU使用情况、日志文件状态

### 2. **实时监控**
```bash
./monitor_training.sh [刷新间隔秒数]
```
- 默认30秒刷新一次
- 也可指定间隔：`./monitor_training.sh 60` (60秒刷新)

### 3. **监控特定信息**

**查看实时日志**:
```bash
tail -f re-orient_training.log
```

**查看GPU状态**:
```bash
watch -n 10 nvidia-smi
```

**查看训练进度**:
```bash
tail -5 training_progress.json
```

**检查训练进程**:
```bash
ps aux | grep re-orient.py
```

## 📊 监控指标说明

### ✅ 正常状态指标
- **Training Process: RUNNING** - 训练进程正常
- **GPU Utilization: 50-100%** - GPU正在工作
- **Memory Usage: 逐渐增加** - 正在使用GPU内存
- **Log entries increasing** - 正在产生训练日志
- **Step X: Reward = Y.ZZZ** - 训练正在进行

### ⚠️ 需要关注的警告
- **Training Process: NOT RUNNING** - 训练停止了
- **GPU Utilization: 0%** - GPU未被使用
- **长时间无日志输出** - 可能卡住了
- **大量Error/ERROR日志** - 出现问题

## 🕐 建议的监控频率

### 初期阶段 (前30分钟)
- 每2-5分钟检查一次快速状态
- JAX初始化可能需要时间

### 正常训练阶段
- 每15-30分钟检查一次
- 使用 `./monitor_training.sh` 自动监控

### 长期运行
- 每小时检查一次
- 设置每日报告提醒

## 🚨 故障排除

### 如果训练停止了：
1. 检查错误日志：`tail -20 re-orient_training.log | grep -i error`
2. 检查GPU状态：`nvidia-smi`
3. 重新启动：`nohup python re-orient.py >> re-orient_training.log 2>&1 &`

### 如果GPU未使用：
1. 确认JAX配置：检查日志中的GPU检测信息
2. 检查CUDA版本：`nvcc --version`
3. 重启可能需要：重启Python进程

## 📈 预期训练时间线

1. **0-10分钟**: JAX初始化和依赖检查
2. **10-30分钟**: 环境加载和参数优化
3. **30分钟后**: 开始实际训练循环
4. **持续**: 实时训练进度更新

记住：JAX GPU初始化需要时间，这是正常的！