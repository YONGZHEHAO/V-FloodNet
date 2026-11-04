# 车辆参考物：人工复核与轮毂几何试算

这条路线复用现有水体/车辆筛查结果，**先人工检查照片，再用标记的轮毂或轮胎边缘与水线做几何计算**。
目前没有训练或接入独立的车轮检测器，不能声称64张车辆图都已自动测出水深。
它使用已有CPU环境的NumPy、OpenCV和Pillow，不需要新增模型或SMPL。

## 当前结果与解释

本地118张输入已全部做过水体分割与目标检测。排除先前人物试跑的7张后，余下64张检出车辆，
包含350个原检测框。目视复核后，两张照片的金属轮毂可以进行探索性拟合，10张保留定性车辆淹没关系，
52张因水线、轮廓、遮挡、像素不足或湿路面反光等原因不估算。这不是车辆检测准确率评估。

两处轮毂淹没高度比例约33.3%和36.9%；它们对标注点和水线较敏感，**不是可靠厘米水深**。
前者在当前标注扰动设定下的5%–95%分位约15.0%–55.0%，后者约23.0%–52.5%。
这些分位仅描述假定像素扰动造成的数值变化，不是实测误差、准确率或置信区间。
照片中的水流、波浪、视角和轮毂边缘模糊带来的系统误差不包含在其中。

`reference_submergence_fraction` 的分母是所选完整参考物的直径：

- `rim`：金属轮毂；绝不能把轮拱、阴影或轮胎外缘混作轮毂边缘。
- `outer_tire`：整条轮胎；本批两例均没有使用这个类型。

该比例是**淹没高度/参考物直径**，不是淹没面积，也不能与人物身高比例直接平均或比较。
对于轮毂参照，若将来核实轮毂直径及轮胎半径，局部竖直几何近似下可由
`h ≈ 轮胎半径 − 轮毂半径 + 轮毂比例 × 轮毂直径` 换算地面至水面的高度。
仍需确认车辆未漂浮/倾斜、车轮正常落地、轮胎变形、路面坡度、相机投影及水线是否合适。
本入口不提供默认车型尺寸，不计算厘米水深，全部 `metric_depth_cm` 保持null。

## 1. 导出车辆候选

先按 `docs/CODEX_CLOUD.md` 生成 `output/recognition/hydrodynamic118/results.json`，然后：

```bash
bash .codex/run.sh scripts/prepare_vehicle_candidates.py --recognition-dir output/recognition/hydrodynamic118 --input-dir assets/hydrodynamic_events_20261103/images --output-dir output/vehicle_candidates/review1
```

可加 `--people-results output/people_depth/local_batch_20261104/batch_results.json` 排除已试跑人体网格的照片。
不传此参数时可检查所有车辆照片，不能把排除规则当作方法本身的限制。
输出 `vehicles.json`、原图预览、各车辆裁剪和分页缩略图；每张源图都核对原筛查记录的SHA256。
检出车辆只是筛查线索，不代表车轮、水线存在或可见。

## 2. 记录人工决策和标记

新建私下保存的 `wheel_annotations.json`，包含两个列表：

1. `image_reviews`：为 `vehicles.json` 中每个候选图给一个 `image_id`、`decision`、`visual_level`、
   `target_description` 和 `reason`。`decision` 取 `geometry_trial`、`qualitative_only` 或 `unavailable`。
2. `wheel_annotations`：只为获准几何试算的图记录所选车轮。每条包括：
   `wheel_id`、`image_id`、`filename`、`source_sha256`、`reference_kind`、`reviewed: true`、
   `boundary_points_xy`、`waterline_xy`、`point_uncertainty_px` 和 `waterline_uncertainty_px`。

坐标必须是**原图像素**，x向右、y向下；不能从展示缩略图直接抄坐标。
标记至少8个可见的同一轮廓边缘点，尽量覆盖足够弧段；不能在被水遮住的轮廓上编造点。
水线用两点定义参考物所在平面中的局部水/物体交界，不能标路面反光、车轮阴影或随意取图像水平线。
当前两例的边缘点扰动标准差取1.5 px，水线端点取3或4 px；这些是分析者设定的敏感性参数，不是标注精度标定值。

可用的 `visual_level` 值：

| 值 | 解释 |
| --- | --- |
| `reference_partial_submergence` | 参考物局部淹没，批准几何试算 |
| `partial_wheel_submergence` | 只能看出车轮局部淹没 |
| `water_reaches_vehicle_body` | 只能看出水至车身下部或更高 |
| `visible_vehicle_water_contact` | 只能确认车辆接触积水，具体等级不清 |
| `unavailable` | 无法可靠判读；不是零水深 |

## 3. 计算与复核

```bash
bash .codex/run.sh scripts/estimate_wheel_depth.py --vehicles output/vehicle_candidates/review1/vehicles.json --annotations output/vehicle_candidates/review1/wheel_annotations.json --input-dir assets/hydrodynamic_events_20261103/images --output-dir output/wheel_depth/run1
```

使用新输出目录。程序拟合可见轮廓的椭圆，沿所标水线法向取得椭圆支撑半径r，令d为椭圆中心到水线的有符号距离，
按 `q = (1 + d/r) / 2` 得到相对高度。该关系假定竖直圆形参考物可作局部仿射投影，不等同于完整透视校正。
可见弧段少于120°、参考物过小/过窄、点不成椭圆、拟合不稳定或水线不穿过参考物时拒绝给数值。
默认进行400次独立高斯像素扰动，固定随机种子，仅对有效试算汇总分位，并记录有效次数。
拟合残差小只说明标记点近似椭圆，不能证明真实轮廓、隐藏部分或水线正确。

输出包括：

- `index.html`：原图、拟合检查图及几何/定性/不可估筛选。
- `results.csv`、`results.json`：每张照片的决策、每个参考物的原始比例和敏感性；多参考物分别记录。
- `coverage.csv`：全部输入照片在本轮的去向，不遗漏未检出车辆或已进入人物路线的照片。
- `annotations.json`：人工原始坐标与判断；`diagnostics/`：可见点、拟合轮廓和水线叠加。

原图、标注和结果保存在Git忽略的 `output/`，不提交模型、账号信息或授权文件。
定性关系来自目视复核，不是新模型分类预测；没有可靠线索的照片保留不可估。
同源连续帧不能视为独立现场观测，照片来源的地点、时间及事件绑定也需单独核实。

## 检查与参考

```bash
bash .codex/run.sh -m unittest discover -s tests -v
```

检查覆盖已知圆几何、仿射斜视一致性、短弧/共线/无交点拒绝、人物批处理和水体掩膜回归。
这些数值检查不是现场水深精度验证。

参考物方法参考 [Gao等：Measuring urban waterlogging depths from video images based on reference objects](https://onlinelibrary.wiley.com/doi/full/10.1111/jfr3.12948)。
本扩展的人工标注、椭圆拟合及敏感性分析为本Fork的探索实现，未复现该论文的训练与验证流程，不引用其精度作为本批照片精度。
