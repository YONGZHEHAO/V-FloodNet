# 人工复核页面

本地离线页面用于核对积水照片、车辆等级候选、人物与轮毂试算。
保留原图和模型结果；人工结论最初全部为待复核。
页面无需联网、安装前端软件或启动服务，不上传照片或结论。

## 开始复核

解压完整复核包后双击 `index.html`，建议用桌面版 Edge 或 Chrome。
不要只移动HTML文件，保留旁边的 `data.js`、`app.js`、`app.css`、`originals/` 和 `evidence/`。

1. 左侧先看“误检提示”和“等级冲突”，再看其他候选和未检出照片。
2. 右侧填写是否可见积水、有无可用参照物；再逐框判断目标、模型等级和相对水线。
3. 看不清时选择“不确定 / 无法判断”，不需要强行指定等级。
4. 可显隐检测框、放大原图、聚焦目标。漏检的参照物可点击“补框”后拖动，再选人物等类型。
5. 完成并下一张，或暂存疑问。J下一张，K上一张，B显隐框，Enter完成；输入框获得焦点时不触发。
6. 经常导出JSON。下次导入JSON恢复，同一张照片的冲突记录只接受更新时间较新的版本。

页面自动暂存在当前浏览器中。换浏览器、移动文件夹、无痕模式或清理缓存可能丢失暂存；
JSON备份才是可交付、可迁移的工作记录。CSV便于Excel查看，不能用来恢复页面状态。
完成后把导出的JSON交给Codex，即可按有效目标和相对淹没关系继续整理数据。
如果浏览器不允许本地暂存，页面会提示导出JSON，仍可使用导入导出保存工作。

## 保留的边界

- L0至L4是模型类别，没有自动换算厘米；等级分数也不是准确率。
- 相对水线是人工视觉判断，不是现场测深。
- 旧人物和轮毂比例仅作诊断参考，两种分母不同，不能直接比较或换算。
- 未检测到目标不表示无水或零水深；无参照物也可如实记录。
- 编辑已完成记录会重新标记为待复核；不把未完成判断当成完成结果。
- JSON导入校验数据版本、照片SHA256、目标编号和补框坐标，全部通过才合并；原始模型结果不从导入文件改写。

## 生成新的复核包

使用已经安装好的基础CPU环境，不增加Python或前端依赖：

```bash
bash .codex/run.sh scripts/build_manual_review.py \
  --results-dir output/vehicle_levels/first_batch \
  --input-dir assets/hydrodynamic_events_20261103/images \
  --output-dir output/manual_review/first_batch
```

`--results-dir` 由 [车辆分级入口](VEHICLE_LEVELS.md) 生成。
目录中的既有 `visual_review.json` 和 `overlap_flags.json` 将作为提示载入，不成为人工答案。
可选 `--recognition-results` 加入来源背景；
`--people-results` 指向人物批次的 `batch_results.json`，
`--wheel-results` 指向车轮批次的 `results.json`，同时复制其诊断图。
这些参数均要求对应的本地结果存在；生成器核对输入SHA256，拒绝覆盖既有复核目录。

数据和复核产物保存在被Git忽略的 `output/manual_review/` 中。
GitHub只保存生成器和网页源码，云端任务需先生成模型结果，再生成复核包并下载完整目录。
