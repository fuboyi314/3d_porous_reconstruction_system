# 基于神经网络的三维多孔介质重构系统

## 项目简介
本项目是一个面向 Windows 桌面环境的科研软件 V1.0，采用 Python 3.11、PySide6 与 PyTorch 实现三维多孔介质结构的参数化重构、基础分析、切片显示、结果导出、日志记录与项目保存/加载。

## 功能边界
### 已实现
- 参数输入与合法性校验
- 演示版条件生成网络推理
- 三维体素结构重构
- 实际孔隙率、连通域、比表面积、配位数、孔隙/固体体素统计
- 三方向切片显示与导出
- 项目保存/加载
- 日志记录与结果导出

### 暂不包含
- 在线训练大型模型
- 高精度物理场求解
- 云端服务部署

## 安装步骤
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方式
```bash
python app.py
```

## 最小可运行版本说明
- 首版 GUI 中的三维显示区采用稳定降级占位方案，确保在没有额外可视化运行环境时软件仍可直接启动。
- 后续若需要真实三维窗口，可在 `src/ui/result_panel.py` 中替换三维占位控件，并在 `src/core/task_controller.py` 中接入可视化数据管线。
- 真实训练模型替换入口位于 `src/model/model_loader.py` 与 `src/model/generator.py`。

## 目录说明
- `app.py`：程序入口
- `src/ui/`：界面层
- `src/core/`：配置、校验、任务调度
- `src/model/`：模型定义、加载、推理
- `src/analysis/`：分析与报告生成
- `src/io/`：项目保存与结果导出
- `logs/`：运行日志
- `outputs/`：导出结果
- `projects/`：项目配置文件
- `tests/`：单元测试


## 演示模式说明
- 如果模型文件不存在或损坏，程序会在 `assets/models/` 下自动生成演示 checkpoint，并继续完成一次可视化重构。
- 当前演示重构会将神经网络输出与多尺度平滑随机场融合，以确保首次启动即可得到可分析、可导出的三维多孔介质样本。
