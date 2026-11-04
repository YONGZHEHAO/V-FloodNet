# 人物估深路线：CPU安装与运行

这里保留原论文的METRO人体网格思路，新增无界面的CPU入口。
**公开权重加载、CPU编码器和裁剪/水线数值检查可独立运行；检查通过本身不代表完成了真实照片的人体网格推理。**
完整推理还需要用户从官网获取SMPL人体文件。没有已知身高时，本入口只允许输出模型估计的
淹没身体比例，厘米水深保持null；不会把上游平均身高175.4 cm当作照片里人物的实测身高。

## 1. 安装可公开下载的部分

先有基础CPU环境，再执行：

```bash
bash .codex/setup.sh
bash .codex/setup_people.sh
bash .codex/run.sh scripts/people_depth.py --check --load-weights
```

这会固定MeshTransformer提交 `412b30142f39e4ac4480a54943993e232697c909`，
初始化其固定的旧版transformers子模块，并在被Git忽略的 `.codex/vendor/MeshTransformer/`
内应用CPU设备补丁。不会改动上游子模块指针或全局torch接口。
新增入口使用模型预测的二维网格顶点，不加载opendr渲染器，也不需要GUI。

公开人体权重约973 MB，来自原作者Hugging Face固定revision，校验SHA256。
完整METRO检查点已包含backbone；使用严格参数加载，因此不额外下载HRNet的513 MB预训练权重。
人体扩展使用 `.codex/requirements-people.lock`；它包含基础模型依赖及旧版代码需要的boto3和chumpy。
可选配置会按其锁文件调整部分传递依赖，仍需 `pip check` 通过。
大模型可能占用数GB内存；CPU速度需在目标环境实测。已在本地Linux/WSL的CPU环境上用用户自己的SMPL文件完成真实照片的完整前向；这不是水深精度验证。

## 2. 获取并导入SMPL（用户自己的许可文件）

在 [SMPLify官网](https://smplify.is.tue.mpg.de/) 注册/登录后进入Downloads，下载
**SMPLIFY_CODE_V2**（常见下载名 `mpips_smplify_public_v2.zip`）。
项目需要的文件在压缩包的 `smplify_public/code/models/` 内，文件名为：

```text
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

只从自己的官方下载包导入，不使用来源不明的转载文件。
本地Windows示例：将ZIP保存到 `G:\CodexData\Workspaces\vfloodnet-cloud-20261103\bootstrap`。
在本项目WSL目录运行：

```bash
bash .codex/run.sh scripts/import_smpl.py --archive /mnt/g/CodexData/Workspaces/vfloodnet-cloud-20261103/bootstrap/mpips_smplify_public_v2.zip
bash .codex/run.sh scripts/people_depth.py --check --load-weights
```

导入脚本只取所需的中性人体文件，保存到被忽略的 `records/smpl/`；不提取整包、不读取账号信息，
不覆盖内容不同的已有模型。文件校验记录保存在 `output/people_depth/smpl-file-info.json`。
官网未提供可用的发布者校验值时，本地SHA256仅用于记录文件身份，不代表发布者认证。
授权文件不能提交到公开GitHub或公开构建产物。登录会话不能替代实际下载的文件。

在Codex云端使用时，同样需要把这个已授权文件放到该云端环境自己的 `records/smpl/`，
或用 `--smpl-path` / `VFLOOD_SMPL_PATH` 指向该环境中的私有文件。
本机G盘路径不会自动出现在云端。许可条件见[SMPLify官方许可](https://smplify.is.tue.mpg.de/license.html)。

## 3. 准备人物候选

先完成 `scripts/recognize_depth.py` 的水体分割及人物检测，然后运行：

```bash
bash .codex/run.sh scripts/prepare_people_candidates.py --recognition-dir output/recognition/hydrodynamic118 --input-dir assets/hydrodynamic_events_20261103/images --output-dir output/people_candidates/review1 --top 20
```

输出 `index.html`、`overview.jpg`、`people.json` 和对齐的原图/掩膜裁剪。
当前阈值仅用于排序：检测分数>=0.9、人物框高>=120 px/宽>=35 px、不触及图像边缘、
框宽高比<=0.65，且下半身邻域有一定模型水体像素。
这些启发式条件不能确认姿态、水线或实际淹没；湿路面、伞、雨衣、弯腰、骑行和其他人物都会干扰。
人工复核后选定一个 `person_id`，逐个试跑，不能把全部人物框自动当作有效水深样本。

## 4. 实际网格推理与水深换算

SMPL就位后，对已目视复核的单个人物运行：

```bash
bash .codex/run.sh scripts/people_depth.py --candidates output/people_candidates/review1/people.json --person-id PERSON_ID --output-dir output/people_depth/run1
```

如已独立确认该人物身高，可增加 `--height-cm`，例如 `--height-cm 170` 只表示参数格式，
不能作为照片人物的默认身高。无此参数时，即使网格推理完成，`metric_depth_cm` 仍是null。
没有有效水/非水顶点边界时，`submerged_body_fraction` 也为null。
本入口默认 `--device cpu`，在具备合适CUDA版torch的环境可显式指定 `--device cuda`。

输出保存在指定目录：

- `PERSON_ID.json`：人物比例、可选厘米水深、来源与裁剪SHA256、相机参数及运行状态。
- `PERSON_ID_diagnostic.png`：对齐的输入图、水体掩膜叠加、预测人体顶点、模板水线。蓝色顶点表示掩膜判水，橙色表示非水，红线是模板中的估计水线。
- `PERSON_ID_points.json`：431个预测二维顶点及逐点掩膜标签，方便复核；超出裁剪图的顶点数量记录在结果的 `evidence` 中。估深沿用上游的边界裁剪规则，超界时应额外检查边缘误判。

必须目视检查顶点是否落在选定人物上、雨衣/靴子/其他人物是否干扰了水线。
模板红线不是原照片中的水平水面线，也不代表地面或绝对高程。
人物网格、淹没比例、厘米估计和实测精度验证是不同阶段，报告应分别说明。
有波浪/水花、倾斜姿态、靴子或厚衣物时，即使存在数值也需检查结果，不能直接作为水动力验证真值。

## 5. 可独立运行的检查

```bash
bash .codex/run.sh -m unittest discover -s tests -v
bash .codex/run.sh scripts/people_depth.py --check --load-weights
```

`encoder_cpu_smoke` 检查小型编码器的CPU计算；`full_checkpoint_loaded` 检查真实公开权重
与完整网络的严格匹配；`mesh_inference_run` 只有实际运行SMPL网格后才为true。
缺少SMPL时，`ready_for_mesh_inference` 为false，不宣称整条人物估深链已通过。
