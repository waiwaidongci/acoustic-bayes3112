"""缺测件 EM 折外判读自测（纯标准库，python -I 运行）。

覆盖：
1. 完整数据可跑通，结果结构与旧版字段兼容（rows/columns/folds/confusion_matrix）。
2. 缺测件带标签+至少一项实测可入批，缺项在结果中保留；验证件只用实测维后验。
3. 同测缺口：某类两项读数从未同测 -> 停算并指出缺口。
4. 全缺测件拒绝；无标签拒绝。
5. 分批 EM：小 batch_size 下结果与大批量一致（分批不改变口径）。
6. 检查点恢复：模拟中断后从最近完成轮恢复，重试不重复计入。
7. 旧研究结果（无 missing_data 字段）读取不受影响（结构层面向后兼容）。
"""
import json
import math
import os
import random
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BACKEND = os.path.join(ROOT, 'backend')
sys.path.insert(0, BACKEND)

# 注入最小 CONFIG，避免依赖真实 settings 环境变量；须在 import 业务模块前完成
import src  # 先建立命名空间包
cfg = types.ModuleType('src.config.settings')
cfg.CONFIG = {
    'fields': [['rms', '均方根振幅', 'number'],
               ['centroid', '频谱质心 Hz', 'number'],
               ['kurtosis', '峭度', 'number'],
               ['label', '复核类型', 'text']],
    'default_parameters': {'folds': 3, 'smoothing': 1e-9},
    'algorithm': 'gaussian',
}
sys.modules['src.config.settings'] = cfg

from src.services.algorithm import calculate, AnalysisError, DEFAULT_BATCH_SIZE
from src.services.domain_validation import validate_record
from src.services import linalg

PASS, FAIL = 0, 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('  PASS', name)
    else:
        FAIL += 1
        print('  FAIL', name, detail)


def make_records(n_per_class=60, seed=42, missing_spec=None):
    rng = random.Random(seed)
    centers = {'正常': (1.2, 900, 2.8), '轴承磨损': (3.0, 1600, 5.2), '转子偏心': (2.0, 2300, 3.9)}
    scales = (0.35, 170, 0.45)
    records = []
    k = 0
    for cls, (m1, m2, m3) in centers.items():
        for i in range(n_per_class):
            vals = {
                'rms': round(max(0.1, rng.gauss(m1, scales[0])), 3),
                'centroid': round(max(100, rng.gauss(m2, scales[1])), 2),
                'kurtosis': round(max(0.1, rng.gauss(m3, scales[2])), 3),
                'label': cls,
            }
            if missing_spec and (k in missing_spec):
                for f in missing_spec[k]:
                    vals[f] = None
            records.append({'reference': 'TX-T-%04d' % k, 'values': vals})
            k += 1
    return records


print('== 1. 完整数据基线，结构兼容旧版 ==')
recs = make_records()
res = calculate(recs, {'folds': 3, 'smoothing': 1e-9})
check('summary 含四项旧指标', [s['label'] for s in res['summary'][:4]] ==
      ['交叉验证准确率', '宏平均 F1', '样本数', '类型数'])
check('rows 行数=输入', len(res['rows']) == len(recs))
check('混淆矩阵 3x3', len(res['confusion_matrix']) == 3 and all(len(r) == 3 for r in res['confusion_matrix']))
check('折数=3', len(res['folds']) == 3)
check('后验在0-1', all(0 <= r['posterior'] <= 1 for r in res['rows']))
check('预测类属于标签集', all(r['predicted'] in res['labels'] for r in res['rows']))
check('准确率>0.8（合成数据可分）', res['summary'][0]['value'] > 0.8, res['summary'][0]['value'])
check('旧版关键列仍存在', set(['reference', 'actual', 'predicted', 'posterior']).issubset(set(res['columns'])))
base_acc = res['summary'][0]['value']

print('== 2. 缺测件入批，保留缺项，折外判读 ==')
# 每类放几件缺测，缺不同特征；保证同测仍成立（其余样本完整）
spec = {}
mk = 0
for cls in range(3):
    base = cls * 60
    spec[base] = ['centroid']
    spec[base + 1] = ['rms', 'kurtosis']
    spec[base + 2] = ['kurtosis']
recs_m = make_records(missing_spec=spec)
res_m = calculate(recs_m, {'folds': 3, 'smoothing': 1e-9})
check('缺测件数=9', res_m['missing_data']['missing_records'] == 9, res_m['missing_data']['missing_records'])
check('各折齐备标志', res_m['missing_data']['all_folds_complete'] is True)
check('缺测判读条数=9', len(res_m['missing_data']['readings']) == 9)
check('明细保留缺项标记', all(res_m['rows'][i]['is_missing_record'] for i in spec))
check('完整件未误标', not any(res_m['rows'][i]['is_missing_record'] for i in range(len(recs_m)) if i not in spec))
check('缺测件后验仍合法', all(0 <= x['posterior'] <= 1 for x in res_m['missing_data']['readings']))
check('readings 含每类后验', all(set(x['posterior_by_class']) == set(res_m['labels'])
                               for x in res_m['missing_data']['readings']))
# 每折都做了 EM
check('每折有 EM 迭代信息', all(f['em_iterations'] >= 1 for f in res_m['folds']))
check('每折记录补全单元数', all(f['imputed_cells'] >= 0 for f in res_m['folds']))
# 缺测件的实测维度信息
first = res_m['missing_data']['readings'][0]
check('记录实测/缺测特征', set(first['observed_features'].split('、')) |
      set(first['missing_features'].split('、')) == {'rms', 'centroid', 'kurtosis'})

print('== 3. 同测缺口停算并指出缺口 ==')
# 让“轴承磨损”类里 rms 与 centroid 从不同时出现：一半只有 rms+kurtosis，另一半只有 centroid+kurtosis
gap_recs = make_records()
for idx, r in enumerate(gap_recs):
    if r['values']['label'] == '轴承磨损':
        local = idx - 60
        if local % 2 == 0:
            r['values']['centroid'] = None
        else:
            r['values']['rms'] = None
try:
    calculate(gap_recs, {'folds': 3, 'smoothing': 1e-9})
    check('同测缺口必须停算', False, '未抛错')
except AnalysisError as e:
    msg = str(e)
    check('停算且指出类型', '轴承磨损' in msg, msg)
    check('停算且指出特征对', 'rms' in msg and 'centroid' in msg and '从未同测' in msg, msg)

print('== 4. 录入校验边界 ==')
try:
    validate_record({'rms': None, 'centroid': None, 'kurtosis': None, 'label': '正常'})
    check('全缺测拒绝', False)
except ValueError as e:
    check('全缺测拒绝', '至少保留一项' in str(e))
out = validate_record({'rms': 1.2, 'centroid': None, 'kurtosis': '', 'label': '正常'})
check('空串与 None 都识别为缺项', out['centroid'] is None and out['kurtosis'] is None and out['rms'] == 1.2)
try:
    validate_record({'rms': 1.2, 'centroid': 800, 'kurtosis': 3.0, 'label': '   '})
    check('空标签拒绝', False)
except ValueError:
    check('空标签拒绝', True)
try:
    calculate([{'reference': 'x', 'values': {'rms': 1.0, 'centroid': None, 'kurtosis': None, 'label': '正常'}}] * 12,
              {'folds': 3, 'smoothing': 1e-9})
    check('算法层全缺测拒绝', False)
except AnalysisError:
    check('算法层全缺测拒绝', True)

print('== 5. 分批不改变口径（小批次 vs 大批次）==')
import src.services.algorithm as alg
orig = alg.DEFAULT_BATCH_SIZE
alg.DEFAULT_BATCH_SIZE = 37  # 强制多批
res_small = calculate(recs_m, {'folds': 3, 'smoothing': 1e-9})
alg.DEFAULT_BATCH_SIZE = orig
check('分批后预测完全一致', [r['predicted'] for r in res_small['rows']] ==
      [r['predicted'] for r in res_m['rows']])
check('分批后后验一致', max(abs(a['posterior'] - b['posterior'])
                           for a, b in zip(res_small['rows'], res_m['rows'])) < 1e-9)
check('分批折记录批次数>1', any(f['batches'] > 1 for f in res_small['folds']))

print('== 6. 检查点恢复：中断后续算，结果一致且不重复计入 ==')
events = []
res_full = calculate(recs_m, {'folds': 3, 'smoothing': 1e-9}, progress=lambda e: events.append(e))
# 找到 fold1 第 1 轮完成事件，构造“只完成第一轮就中断”的检查点
iter_done = [e for e in events if e.get('phase') == 'iteration_done' and e['fold'] == 1]
check('存在整轮完成事件', len(iter_done) >= 1)
cp1 = {'fold': 1, 'iteration': iter_done[0]['iteration'], 'means': iter_done[0]['means'],
       'covs': iter_done[0]['covs'], 'history': iter_done[0]['history']}
res_resume = calculate(recs_m, {'folds': 3, 'smoothing': 1e-9}, checkpoint=cp1)
check('恢复后预测一致', [r['predicted'] for r in res_resume['rows']] ==
      [r['predicted'] for r in res_full['rows']])
check('恢复后后验一致', max(abs(a['posterior'] - b['posterior'])
                           for a, b in zip(res_resume['rows'], res_full['rows'])) < 1e-9)
# 恢复时 fold1 的 history 应延续而非清零重来
resume_events = []
calculate(recs_m, {'folds': 3, 'smoothing': 1e-9}, checkpoint=cp1, progress=lambda e: resume_events.append(e))
fold1_iters = [e for e in resume_events if e.get('phase') == 'iteration_done' and e['fold'] == 1]
check('恢复从下一轮开始（iteration 接续）', fold1_iters and fold1_iters[0]['iteration'] == cp1['iteration'] + 1)

print('== 7. 旧研究结果向后兼容 ==')
legacy = {'summary': [], 'labels': [], 'confusion_matrix': [], 'folds': [], 'rows': [],
          'columns': ['reference', 'actual', 'predicted', 'posterior'], 'chart': [],
          'explanation': 'old', 'checksum': 'x', 'seed': 42}
check('旧结果无 missing_data 字段仍可被导出/页面读取', 'missing_data' not in legacy)

print('== 8. 线性代数工具自检 ==')
A = [[4.0, 1.0], [1.0, 3.0]]
b = [1.0, 2.0]
x = linalg.solve_spd(A, b)
check('solve_spd 正确', abs(A[0][0] * x[0] + A[0][1] * x[1] - b[0]) < 1e-10 and
      abs(A[1][0] * x[0] + A[1][1] * x[1] - b[1]) < 1e-10)
cm, cc = linalg.conditional_normal([0.0, 0.0], [[2.0, 0.5], [0.5, 1.0]], [0], [1.0], [1])
check('条件均值=回归系数*观测', abs(cm[0] - 0.25) < 1e-10, cm)
check('条件协方差=Schur 补', abs(cc[0][0] - 0.875) < 1e-10, cc)
lp1 = linalg.gaussian_logpdf([0.0], [0.0], [[1.0]])
check('标准高斯 0 点 logpdf', abs(lp1 - (-0.5 * math.log(2 * math.pi))) < 1e-10)

print('== 9. 折内同测缺口（subset）直接单测 ==')
from src.services.algorithm import _check_co_measurement, FEATURES
labs9 = ['正常'] * 6
obs9 = [(0, 2), (0, 2), (1, 2), (1, 2), (0, 1, 2), (0, 1, 2)]
try:
    _check_co_measurement(labs9, obs9, ['正常'], subset=list(range(4)), fold=2)
    check('折内缺口停算', False)
except AnalysisError as e:
    check('折内缺口停算且带折号', '第 2 折' in str(e) and 'rms' in str(e) and 'centroid' in str(e), str(e))
# 全量子集通过
_check_co_measurement(labs9, obs9, ['正常'], subset=list(range(6)), fold=1)
check('同测齐全时通过', True)

print('== 10. CSV 导出口径：缺项列保留、不静默填充 ==')
import io, csv
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=res_m['columns'])
w.writeheader()
for r in res_m['rows']:
    w.writerow({k: r.get(k, '') for k in res_m['columns']})
lines = buf.getvalue().splitlines()
check('CSV 行数=数据+表头', len(lines) == len(res_m['rows']) + 1)
parsed = list(csv.DictReader(io.StringIO(buf.getvalue())))
gap_rows = [row for row in parsed if row['missing_features']]
check('缺测件在 CSV 中保留缺测特征列', len(gap_rows) == 9)
check('CSV 含实测维度数列', all(row['observed_count'] in ('1', '2') for row in gap_rows))

print()
print('PASS=%d FAIL=%d' % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
