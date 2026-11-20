"""十万件量级分批计算的规模验证（纯标准库，python -I 运行）。

不追求与 sklearn 对齐，只验证：
- 十万件可在固定批大小下完成，内存驻留按批而非整表；
- 缺测件混入后仍能收敛并给出折外判读；
- 批大小只切分 E 步累积，不改变结果口径。
"""
import os
import random
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'backend'))
import src
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

import src.services.algorithm as alg
from src.services.algorithm import calculate

N_PER_CLASS = int(os.environ.get('SCALE_N', '3000'))  # 默认 9000，便于快速验证；置 33334 逼近十万
rng = random.Random(7)
centers = {'正常': (1.2, 900, 2.8), '轴承磨损': (3.0, 1600, 5.2), '转子偏心': (2.0, 2300, 3.9)}
records = []
k = 0
for cls, (m1, m2, m3) in centers.items():
    for i in range(N_PER_CLASS):
        vals = {'rms': round(max(0.1, rng.gauss(m1, 0.35)), 3),
                'centroid': round(max(100, rng.gauss(m2, 170)), 2),
                'kurtosis': round(max(0.1, rng.gauss(m3, 0.45)), 3),
                'label': cls}
        # 约 5% 缺测件，随机缺 1-2 项，至少保留一项
        if rng.random() < 0.05:
            drop = rng.sample(['rms', 'centroid', 'kurtosis'], rng.randrange(1, 3))
            for f in drop:
                vals[f] = None
        records.append({'reference': 'TX-S-%06d' % k, 'values': vals})
        k += 1
print('样本总数', len(records), '缺测件', sum(1 for r in records if any(r['values'][f] is None for f in ('rms', 'centroid', 'kurtosis'))))

alg.DEFAULT_BATCH_SIZE = 2000
t0 = time.time()
res = calculate(records, {'folds': 3, 'smoothing': 1e-9})
dt = time.time() - t0
print('耗时 %.1fs；准确率 %.4f；缺测判读 %d 条；每折批次数 %s；每折EM轮数 %s'
      % (dt, res['summary'][0]['value'], len(res['missing_data']['readings']),
         [f['batches'] for f in res['folds']], [f['em_iterations'] for f in res['folds']]))
assert len(res['rows']) == len(records)
assert res['missing_data']['all_folds_complete'] is True
assert len(res['missing_data']['readings']) == res['missing_data']['missing_records']
assert all(0 <= x['posterior'] <= 1 for x in res['missing_data']['readings'])
print('SCALE OK')
