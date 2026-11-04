# FLOOD-DEPTH-ML 车辆淹没等级试跑

这条可选路线使用 [mayankmi/FLOOD-DEPTH-ML](https://github.com/mayankmi/FLOOD-DEPTH-ML)
公开的 `best_car.pt`，在整张图片中检测车辆并预测 `Level 0` 到 `Level 4`。
它与原 V-FloodNet 的分割、人物网格和人工车轮几何路线分别运行。
等级属于模型预测，不是已验证的现场水深，不把标签转为厘米或淹没百分比。

## 安装与模型来源

先按 [基础环境说明](CODEX_CLOUD.md) 安装 Linux CPU 环境，再执行：

```bash
bash .codex/setup_vehicle_levels.sh
```

可在本地 WSL2 或已安装基础环境的 Linux 云端运行。
安装脚本将五个固定版本的附加包安装到 `.codex/vendor/vehicle-levels/`，
只由本入口加入导入路径，不更改 `.venv`、基础 CPU 锁文件或人物路线依赖。
每个下载包核验 PyPI SHA256；现有 PyTorch 2.2.2 CPU、NumPy 1.23.5
与唯一的 contrib-headless OpenCV 继续使用。
附加包依赖逐项检查；`opencv-python` 的功能由基础 headless OpenCV 提供，避免重复安装 `cv2`。
这不是不带基础环境的独立完整依赖锁。

权重固定为上游提交 `1c05473c537423b565051d6fec95dfdfeda05560`，
SHA256 `0b15b6114ba663d98509331faac467a1c838fb1f2c1235dfcf18fd63395b62ee`。
检查点记录的 Ultralytics 版本是 `8.3.133`，入口使用该版本。
权重保存到忽略目录 `records/vehicle_levels/`，不复制到 Git。
运行不需要 Streamlit、摄像头或网页服务，也不上传输入图片。

## 运行

使用已有 V-FloodNet 筛查结果以保留图片来源和 SHA256：

```bash
bash .codex/run.sh scripts/vehicle_levels.py \
  --input-dir assets/hydrodynamic_events_20261103/images \
  --recognition-results output/recognition/hydrodynamic118/results.json \
  --output-dir output/vehicle_levels/first_batch
```

如果旧结果在其他目录，把 `--recognition-results` 换成实际路径。
可选 `--wheel-results output/wheel_depth/final_20261104/results.json` 关联旧车轮复核理由；
它不影响模型输出，也不提供等级真值。
用重复的 `--image-id 图片ID` 可选择试跑子集。每次使用新输出目录，保留既有结果。

输出 `index.html` 逐图查看带框图片；`results.csv` 按图片列出候选，
`detections.csv` 按目标列出等级、置信度、检测框匹配情况，
`results.json` 保留模型、软件、输入与脚本身份及参数。

生成概览图、重叠框筛查和来源文件检查：

```bash
bash .codex/run.sh scripts/review_vehicle_levels.py \
  --results-dir output/vehicle_levels/first_batch \
  --input-dir assets/hydrodynamic_events_20261103/images
```

`review.html` 为复核入口。可选 `--reviews visual_review.json` 载入人工复核记录，
格式见脚本；没有提供记录的图片保持未逐目标复核。
不同等级框存在较大重叠时只打标记，不自动合并，也不推断哪一个等级正确。

## 如何理解结果

- 默认保留置信度不低于 0.25 的目标，以便检查小目标与误检。
- `matched_vehicle_candidate`：至少有一个目标置信度不低于 0.50，且与旧汽车框的 IoU 不低于 0.35。
  这只说明两个模型在汽车位置上相符，不验证淹没等级。
- `low_confidence_only`：只有低于 0.50 的目标；暂不接受为稳定输出。
- `unmatched_model_candidate`：有不低于 0.50 的目标，但未匹配旧汽车框，需要排查新检出与误检。
- `no_detection`：未检出超过阈值的目标。不能解释成无水、无车或零水深。

汽车训练类别不能自动推广为摩托车、公交车或卡车的水深标尺。
同一张图片里的车辆可能处于不同位置、地面高程和水深，不能把某个等级代表整张图片。
远处车辆、车辆下部被遮挡、车棚、反光、水花、夜景和低分辨率均可能造成误判。
置信度不是在这118张照片上的准确率。人工检查目标与可见水线后也仍缺少实测深度真值。
所有 `estimated_depth_cm` 保持空，`accuracy_validated` 保持 `false`。

上游论文：[FLOOD-DEPTH-ML, Results in Engineering](https://doi.org/10.1016/j.rineng.2026.109495)。
论文按车型概化的深度区间不能直接成为本批照片的厘米测量。

## 2026-11-04 本批试跑

本地 WSL CPU 已处理118张来源照片（不是云端账号已运行的证据）：
63张有模型候选，55张无检测，合计170个框。47张至少有一个高于筛查阈值且
匹配旧汽车框的候选，14张仅低置信候选，2张有高置信但未匹配旧汽车框的候选。
这些是检测统计，不能表述成63张或47张已成功识别水深。

全批概览检查后选取16张查看较大原图，发现人物、花盆、桶、盘卷材料和地面被误认成车辆，
也有重复异级框、遮挡和疑似等级低估。旧检测器同样会误检电缆盘，
因此两个模型在位置上相符也不能验证车辆身份。
这16张是针对问题选取的示例，不能计算整个数据集的误检率或水深准确率。

本批建议仅把该模型用于候选筛选与人工复核；未输出任何已验证厘米水深。
数据、完整预测和复核图片保存在本地 `output/vehicle_levels/batch118_20261104/`，
不随代码提交 GitHub，云端需重新运行后生成。
