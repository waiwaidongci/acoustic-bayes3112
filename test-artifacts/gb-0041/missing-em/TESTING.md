# 缺测件 EM 折外判读测试

纯 Python 标准库实现与自测，运行环境无需 numpy/sklearn，统一用隔离模式 `python -I` 运行，避免当前目录下的同名模块被导入。

## 运行

```bash
# 功能与口径自测（41 项断言）
python3 -I test_algorithm.py

# 规模验证：默认 9000 条；置 SCALE_N=33334 即约十万件
python3 -I test_scale.py
SCALE_N=33334 python3 -I test_scale.py
```

## test_algorithm.py 覆盖

1. 完整数据基线：结果结构与旧版字段兼容（summary/rows/columns/folds/confusion_matrix/chart/checksum/seed）。
2. 缺测件入批：带故障标签与至少一项实测值，缺项在结果中保留（`missing_features`、`is_missing_record`、`observed_count`）；各折齐备后 `missing_data.readings` 才给出折外判读与每类后验。
3. 同测缺口：某类两项读数从未同测时停算，错误信息指出类型与特征对；折内 subset 检查带折号。
4. 录入校验：全缺测拒绝、空串与 `null` 均识别为缺项、空标签拒绝、算法层全缺测拒绝。
5. 分批口径：把批大小强制改为 37 后，预测与后验和默认批大小逐元素一致，证明分批只切分 E 步累积。
6. 检查点恢复：用“第 1 折第 1 轮完成”的检查点续算，最终预测/后验与一次跑通完全一致，且下一轮 iteration 接续（不重头计数、不重复计入）。
7. 旧研究向后兼容：无 `missing_data` 字段的历史结果可被页面与 CSV 导出读取。
8. 线性代数工具：正定方程求解、条件高斯均值/Schur 补、标准高斯 logpdf。
9. CSV 导出：缺测特征列与实测维度数列在导出中保留。

## test_scale.py 覆盖

- 固定批大小下流式累积，十万件（100002 行、约 5% 缺测件）可完成；输出行数等于输入，缺测判读条数等于缺测件数，后验合法。
- 参考耗时（受限容器）：约 5 秒、34 批/折、EM 4 轮收敛；tracemalloc 峰值约 106 MB（主要为输入快照本身，E 步统计量按批驻留）。
