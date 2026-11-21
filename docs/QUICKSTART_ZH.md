# RD-Fuse：使用说明

## 文件入口

- `paper/Manuscript.pdf`：原样保存的正式论文 PDF（10 页）。
- `paper/Cover_Letter.pdf`：原样保存的投稿信 PDF（2 页）。
- `paper/latex/`：根据正式 PDF 重建的可编辑 LaTeX，并非原始 Word 排版源。若投稿系统要求源文件与 PDF 完全一致，请以作者实际生成该 PDF 的源文件为准。
- `rd_fuse/`：模型结构、语义探测、效用预测、联合路由、检测头及训练相关实现。
- `configs/`、`scripts/`、`tests/`：运行配置、训练/评测脚本及测试。

## 环境和基础测试

```bash
python -m pip install -e '.[test]'
python -m pytest -q
bash scripts/run_smoke.sh
```

`run_smoke.sh` 生成的是**合成测试数据**，只用于检查代码能否顺利执行，不是 M3FD、LLVIP 上的实验。真实论文精度和 Jetson 性能数据保留在原始论文里，本压缩包没有包含原始训练权重、数据集及 TensorRT 原始测量记录，不能用烟雾测试替代这些记录。

## 与论文相应的功能

- `rd_fuse/backbones.py`：RGB / Thermal 双流与按路由选择的深层执行。
- `rd_fuse/probe.py`：语义状态、模态可靠性与对齐损失。
- `rd_fuse/routing.py`：全局/局部效用预测和预算约束选择。
- `rd_fuse/training.py`：强制路径、反事实单窗口监督、多窗口联合损失 Regret 阶段。
- `rd_fuse/model.py`：按需执行与选中窗口交互。
- `scripts/profile_pytorch.py`：PyTorch 时延评估（不是 TensorRT 测量）。

## 真正的实验与投稿

论文 PDF 与 Cover Letter 未作任何内容修改。若你已保存了真实实验数据和原始投稿源文件，请以那些作者记录为准；这份代码属于从论文描述构建的实现，未与原始实验代码做逐项一致性验证。投稿系统要求上传哪些材料，以其页面实际要求为准。

## 最终补充配置

`configs/m3fd.yaml` 与 `configs/llvip.yaml` 提供按数据集区分的三阶段参考运行设置。`configs/author_confirmed_profile.json` 保存作者确认的路由成本（共享 5 ms、Tref 30 ms、预算 21/25.5/30 ms）；`docs/AUTHOR_CONFIRMED_SETTINGS.md` 与 `docs/author_confirmed_seed_summary.csv` 保存确认的补充参数和逐种子汇总。上述确认值未附原始实验运行日志或硬件测量记录，不作为独立复现证据。

三阶段启动、余弦调度、反事实目标缓存刷新、多预算、温度退火、同步翻转及逐样本路由输出已接入参考实现。实际训练使用原始数据和设备成本；可选 COCO 评价接口只使用 manifest 中已有标注，不恢复缺失的 crowd/ignore 信息。详见 README 中的命令。正式 PDF 保持不变。
