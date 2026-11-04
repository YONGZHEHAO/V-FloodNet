# 在 Codex 云端运行 V-FloodNet

本分支为 **Linux x86_64、Python 3.10.16、PyTorch 2.2.2 CPU** 提供安装脚本。
环境文件保存在 GitHub；实际安装发生在 Codex 创建的云端机器中。
仓库文件不会自动创建或发布你账号内的 Codex 云端环境。

## 创建云端环境

在 Codex 的 **Work in / 工作位置 → Cloud / 云端 → Create environment / 创建环境**
中选择 `YONGZHEHAO/V-FloodNet` 的 `main` 分支，要求安装阶段执行：

```bash
bash .codex/setup.sh
```

安装后执行下面的检查，再发布环境并创建新任务：

```bash
bash .codex/run.sh .codex/check_environment.py --profile cpu --require-weights --smoke
```

如果你的界面仍使用旧版环境设置，将同一条安装命令放进 **Setup script**。
不依赖 `conda activate` 或上一个终端的 `export`；运行时统一使用 `.codex/run.sh`。
云端环境的创建、安装和发布流程见
[OpenAI 官方说明](https://learn.chatgpt.com/docs/environments/cloud-environments)。

可以直接把这段话发给云端 Codex：

> 阅读 AGENTS.md 和 docs/CODEX_CLOUD.md，执行 bash .codex/setup.sh，再运行
> bash .codex/run.sh .codex/check_environment.py --profile cpu --require-weights --smoke。
> 检查环境报告和输出图片。先复现图片水体分割，再检查输入照片是否满足水深估计的参照物条件。
> 遇到无 GPU、缺少模型或没有参考物体时，说明具体限制，不要编造水深。

## 安装内容与选项

| 配置 | 内容 | 命令 |
| --- | --- | --- |
| 默认 `cpu` | 图片水体分割、Detectron2 CPU、停止标志水深估计及官方权重 | `bash .codex/setup.sh` |
| `image` | 仅图片水体分割及其官方权重；跳过 Detectron2 编译和检测权重 | `VFLOOD_PROFILE=image bash .codex/setup.sh` |
| 仅安装软件 | 不下载权重；只能检查软件，不能运行真实推理 | `VFLOOD_DOWNLOAD_WEIGHTS=0 bash .codex/setup.sh` |

默认安装会编译 Detectron2，需要 `git`、`g++` / `build-essential`。
编译并发默认限制为 2；内存不足时可设 `MAX_JOBS=1`，不要忽略编译错误。
下载包含约 803 MB 的官方 `records.zip` 和约 495 MB 的检测权重，另需环境、编译及缓存空间。
不会下载完整训练集或从头训练。权重和虚拟环境均不会提交 GitHub。
`.codex/cache/`、`.codex/python/`、`.codex/tmp/` 保存下载、工具链与临时文件。
`VFLOOD_CACHE_DIR` 可指定模型压缩包缓存位置。

网络策略至少需要放行 PyPI / 文件下载域、`download.pytorch.org`、GitHub / Release
下载域、Hugging Face / 重定向下载域，以及 `dl.fbaipublicfiles.com`。
下载应在环境安装阶段完成；若收到域名拒绝，核对失败请求的实际主机并在环境中配置后重试。
不要关闭 TLS 验证。修改依赖后应重新安装并重新发布环境。

旧版依赖文件仍保留作上游记录。云端使用 `.codex/requirements-cpu.lock`，避免
同时安装多个共享 `cv2` 命名空间的 OpenCV 包，也避免自动升级 PyTorch 后改变旧检查点行为。
官方模型使用 Python 对象序列化，只加载可信来源；下载脚本核验模型文件的 SHA256。

## 图片分割

### 一条命令开始识别本仓库的118张照片

在环境已安装后，从仓库根目录执行：

```bash
bash .codex/run.sh scripts/recognize_depth.py --input-dir assets/hydrodynamic_events_20261103/images --manifest assets/hydrodynamic_events_20261103/manifest.json --test-name hydrodynamic118
```

此入口同时执行水体分割和人物、车辆、停止标志的检测，输出到
`output/recognition/hydrodynamic118/`。打开 `index.html` 逐图查看，
`results.json` 保留来源、输入SHA256、模型检测结果和水深状态。
这里的检测仅用来筛选参照物：检出人物或汽车，不代表已完成人物或车辆估深。
这118张输入当前没有已核实参考尺寸，因此不能自动填入厘米水深。
没有停止标志、没有水像素或缺尺寸，分别记录原因并保留 `estimated_depth_cm: null`。

若使用自己的平铺图片目录，将 `--input-dir` 换成该目录，去掉不对应的 `--manifest`。
已核实同批照片的停止标志尺寸时，可以同时提供 `--sign-size-cm` 和
`--pole-height-cm`；图中必须是这些尺寸适用的同类参考物。
新的一次运行使用新的 `--test-name`，脚本会拒绝覆盖旧目录。

复现项目自带的8张停止标志图片：

```bash
bash .codex/run.sh scripts/recognize_depth.py --input-dir assets/lake_exp --test-name lake_demo --demo
```

只有项目自带的 `assets/lake_exp` 允许 `--demo` 使用上游默认尺寸。
演示结果仍属于模型估计，不是水深真值。
也可在 GitHub Actions 中手动运行 **Recognize water depth samples** 工作流，
它会执行掩膜回归检查、8张演示图和118张来源样本，下载产物后打开其中的 `index.html`。
这些脚本也能在已经安装好的 Codex 云端环境运行；GitHub Actions 是独立的 Linux 运行环境。

### 单独运行图片分割

```bash
bash .codex/run.sh test_image_seg.py --test-path assets/img_exp --test-name demo
```

自己的 jpg/png 图片放到 `WaterDataset/my_images/` 等被忽略的目录：

```bash
bash .codex/run.sh test_image_seg.py --test-path WaterDataset/my_images --test-name my_images
```

输出：`output/segs/<test-name>/mask/` 与 `output/segs/<test-name>/overlay/`。
这一步只产生水体掩膜，不能单独给出水深。

## 停止标志参考物水深估计（CPU）

先用同一批图片和同一个 `test-name` 完成分割，再运行：

```bash
bash .codex/run.sh est_waterlevel.py --test-path WaterDataset/my_images --test-name my_images --opt stopsign --device cpu
```

如果知道实物尺寸，显式传入厘米值，例如下面只是**参数格式示例**：

```bash
bash .codex/run.sh est_waterlevel.py --test-path WaterDataset/my_images --test-name my_images --opt stopsign --device cpu --sign-size-cm 79 --pole-height-cm 215.9
```

上游默认板面尺寸为 79 cm、杆高为 215.9 cm。不同国家、道路、安装方式的尺寸可能不同，
不能把默认值当作现场测量。相机透视、遮挡、倾斜、分割误差和参照物附近地面高程也会影响结果。
该方法要求可识别的八边形停止标志及合适视角；普通路牌或任意积水照片不能直接替代。

结果位于 `output/waterlevel/<test-name>_stopsign/`。
`waterdepth.txt` 的列为图片名、淹没比例和**厘米水深**。`-1` 是上游的未估出标记，
不可当作负水深或零水深。演示图片推理完成也不代表已验证真实水深精度。

## 其他路线的边界

新增的人物CPU适配、公开权重检查与SMPL导入步骤见[人物路线说明](PEOPLE_DEPTH.md)。
车辆候选筛查和人工轮毂几何试算见[车辆参考物说明](WHEEL_DEPTH.md)，无需新增依赖或模型；缺尺寸不输出厘米水深。
该扩展与下表的基础CPU配置分开安装；缺少SMPL或没有实际前向结果时不能宣称人物估深已跑通。

| 路线 | 当前 CPU 配置 |
| --- | --- |
| 图片分割 | 提供安装、官方权重和真实图片检查 |
| 停止标志估深 | 提供 Detectron2 CPU、官方检测权重和 CPU 入口 |
| 人物估深 | 可选 `scripts/people_depth.py` CPU入口；另运行 `.codex/setup_people.sh` 并导入自己的SMPL授权文件；原 `--opt people` 入口仍要求CUDA |
| 视频分割 | 未配置；原始入口强制 CUDA，不能承诺在普通 CPU 云端运行 |
| 参考物跟踪 `ref` | 需相机校正矩阵、参考框及物理标定文件；无界面环境不能弹窗选点 |

官方 Codex 文档未承诺本账号环境的 GPU；实际能力应在云端检查，不能仅凭套餐名称推断。
若最终照片以人物为参照，可使用上述可选CPU入口；运行速度取决于目标机器和存储性能。
SMPL 必须按官方许可自行获取，不应上传到公开仓库。

## 验证与故障定位

```bash
bash .codex/run.sh -m pip check
bash .codex/run.sh .codex/check_environment.py --profile cpu --require-weights --smoke
```

检查报告保存为 `.codex/environment-report.json`，示例分割保存为 `output/segs/cloud_smoke/`。
软件检查会加载真实 CPU C++ 算子、PointRend 配置与项目模块；`--smoke` 额外用官方水体
权重处理一张仓库照片，检查掩膜尺寸与标签。它不验证水深精度。

还可增加 `--depth-smoke`，在 `assets/lake_exp/0.jpg` 上实际运行停止标志 CPU 估深，
结果保存为 `output/waterlevel/cloud_stopsign_stopsign/waterdepth.txt`。此检查使用上游默认
实物尺寸，只检验整条推理链能执行，不是实测精度验证。GitHub Actions 会运行这两项检查。

若下载不完整或校验不符，修复网络后重新执行安装，不要跳过校验。
若空间不足，先检查 `.codex/cache/` 的下载和编译缓存；不要删除原始输入数据。
新环境先安装再推理，使用 `image` 配置时检查参数也要改为 `--profile image`。

来源：[原项目](https://github.com/xmlyqing00/V-FloodNet)、
[官方模型](https://huggingface.co/xmlyqing00/V-FloodNet/tree/a0b9c2de193de188434220949f10da84e61844cb)、
[PyTorch 版本安装说明](https://pytorch.org/get-started/previous-versions/)、
[Detectron2 固定源码](https://github.com/facebookresearch/detectron2/tree/247691a1c292654867deb35cabb27f18d75a1363)。
