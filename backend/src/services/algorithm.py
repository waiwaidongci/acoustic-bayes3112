"""高斯类条件模型 + 分层折外验证，支持缺测件。

口径（与研究页面说明一致）：
- 缺测件同样入批：保留故障标签与至少一项实测值，缺项以 None 保留不填。
- 冻结分层折：折划分只依据标签，在任何参数迭代之前固定。
- 每折仅用训练组的实测维度估计各类均值与（全）协方差；
  缺项按类条件高斯分布做 EM 迭代补全（E 步条件均值/二阶矩，M 步充分统计量）。
- 验证件不参与拟合，只在其实测维度上计算各类后验。
- 十万件量级按固定批大小流式累积充分统计量，单批驻留内存；
  每轮各批共用上一轮收敛前的参数。检查点按轮落盘，中断后从最近完成轮恢复，
  重试不会把同一批重复计入。
- 若某一类有任意两项读数从未在同一件上同时实测，则该类协方差对应位置无法
  估计，直接停算并指出缺口特征对。
"""
import hashlib
import json
import math

from .domain_validation import validate_parameters
from .linalg import (
    cholesky,
    conditional_normal,
    gaussian_logpdf,
    mat_zeros,
)

FEATURES = ['rms', 'centroid', 'kurtosis']
# 有限内存：单批最多驻留的样本行数，十万件按此切分逐批累积
DEFAULT_BATCH_SIZE = 5000
EM_MAX_ITERS = 50
EM_TOL = 1e-6
EPS_LOG = -1e18


class AnalysisError(ValueError):
    """带业务语义的可恢复/可展示错误。"""


def _parse_records(records):
    n_features = len(FEATURES)
    refs, labels, observed, values = [], [], [], []
    for r in records:
        data = r['values']
        label = data.get('label')
        if not isinstance(label, str) or not label.strip():
            raise AnalysisError('复核类型不能为空，缺测件也必须带故障标签')
        obs, row_vals = [], [None] * n_features
        for j, f in enumerate(FEATURES):
            v = data.get(f)
            if v is None:
                continue
            if isinstance(v, bool):
                raise AnalysisError('声学特征必须为数值或留空')
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise AnalysisError('声学特征必须为数值或留空')
            if not math.isfinite(v):
                raise AnalysisError('声学特征必须为有限数值')
            if v < 0:
                raise AnalysisError('声学特征不可为负')
            row_vals[j] = v
            obs.append(j)
        if not obs:
            raise AnalysisError('缺测件至少保留一项实测声学特征，样本 %s 全部缺测' % r.get('reference'))
        refs.append(r['reference'])
        labels.append(label.strip())
        observed.append(tuple(obs))
        values.append(row_vals)
    return refs, labels, observed, values


def _check_co_measurement(labels, observed, classes, subset=None, fold=None):
    """同测缺口预检：某类任意两特征从未同时实测则停算。

    subset 给定时只检查该索引集合（用于每折训练组），报错附带折号。
    """
    d = len(FEATURES)
    gaps = []
    for c in classes:
        pair_seen = [[False] * d for _ in range(d)]
        for i in (subset if subset is not None else range(len(labels))):
            if labels[i] != c:
                continue
            for a in observed[i]:
                for b in observed[i]:
                    pair_seen[a][b] = True
        for i in range(d):
            for j in range(i + 1, d):
                if not pair_seen[i][j]:
                    gaps.append({'label': c, 'pair': [FEATURES[i], FEATURES[j]]})
    if gaps:
        where = '（第 %d 折训练组）' % fold if fold else ''
        detail = '；'.join('类型「%s」的 %s 与 %s 从未同测' % (g['label'], g['pair'][0], g['pair'][1]) for g in gaps)
        raise AnalysisError('存在同测缺口，无法估计类协方差，已停算' + where + '：' + detail
                            + '。请补充同时含这两项读数的样本后重试。')
    return gaps


def _stratified_folds(labels, n_folds, seed=42):
    """确定性分层 K 折：类内固定洗牌后按序轮转分配折号，保证可复现。"""
    fold_of = [-1] * len(labels)

    def rnd():
        # 确定性 LCG，仅用于折内洗牌
        rnd.state = (rnd.state * 1103515245 + 12345) & 0x7fffffff
        return rnd.state

    rnd.state = seed
    classes = sorted(set(labels))
    n = len(labels)
    for c in classes:
        idx = [i for i, lab in enumerate(labels) if lab == c]
        # Fisher–Yates 洗牌（确定性）
        for k in range(len(idx) - 1, 0, -1):
            j = rnd() % (k + 1)
            idx[k], idx[j] = idx[j], idx[k]
        for k, i in enumerate(idx):
            fold_of[i] = k % n_folds
    folds = []
    for f in range(n_folds):
        test = [i for i in range(n) if fold_of[i] == f]
        train = [i for i in range(n) if fold_of[i] != f]
        folds.append((train, test))
    return folds


def _regularize(cov, smoothing):
    d = len(cov)
    scale = max(max(cov[i][i] for i in range(d)), 1e-12)
    ridge = smoothing * scale
    for i in range(d):
        cov[i][i] += ridge
    return cov, ridge


def _initial_means(values, observed, classes, class_idx):
    d = len(FEATURES)
    means = {}
    for ci, c in enumerate(classes):
        m = [0.0] * d
        for j in range(d):
            nums = [values[i][j] for i in class_idx[ci] if j in observed[i]]
            if not nums:
                # 该类整列缺测：退回全局该维均值，后面同测/单列检查会兜底
                allnums = [values[i][j] for i in range(len(values)) if j in observed[i]]
                m[j] = sum(allnums) / len(allnums) if allnums else 0.0
            else:
                m[j] = sum(nums) / len(nums)
        means[c] = m
    return means


def _initial_cov(values, observed, classes, class_idx, means):
    d = len(FEATURES)
    covs = {}
    for ci, c in enumerate(classes):
        cov = mat_zeros(d, d)
        for a in range(d):
            for b in range(a, d):
                pts = [(values[i][a], values[i][b]) for i in class_idx[ci]
                       if a in observed[i] and b in observed[i]]
                if pts:
                    ma = sum(p[0] for p in pts) / len(pts)
                    mb = sum(p[1] for p in pts) / len(pts)
                    v = sum((p[0] - ma) * (p[1] - mb) for p in pts) / max(len(pts) - 1, 1)
                else:
                    v = 0.0
                cov[a][b] = cov[b][a] = v
        for a in range(d):
            if cov[a][a] <= 0:
                cov[a][a] = 1e-6
        covs[c] = cov
    return covs


def _accumulate_batch(batch, params, classes, class_of):
    """对一个批次执行一次 E 步并累积充分统计量（内存只驻留当前批）。

    返回每类的 N、向量和、二阶矩和；二阶矩对缺测维使用条件二阶矩
    E[z z^T] = cond_mean cond_mean^T + cond_cov。
    """
    d = len(FEATURES)
    means, covs = params
    stats = {c: {'n': 0, 's1': [0.0] * d, 's2': mat_zeros(d, d)} for c in classes}
    imputations = 0
    for i in batch:
        c = classes[class_of[i]]
        obs = observed_global[i]
        miss = tuple(j for j in range(d) if j not in obs)
        vals = values_global[i]
        st = stats[c]
        st['n'] += 1
        z = vals[:]
        cond_mean, cond_cov = [], []
        if miss:
            cond_mean, cond_cov = conditional_normal(
                means[c], covs[c], list(obs), [vals[j] for j in obs], list(miss))
            for k, j in enumerate(miss):
                z[j] = cond_mean[k]
            imputations += 1
        for a in range(d):
            st['s1'][a] += z[a]
            for b in range(a, d):
                if a in obs and b in obs:
                    moment = vals[a] * vals[b]
                elif a in obs:  # b 缺测
                    k = miss.index(b)
                    moment = vals[a] * cond_mean[k]
                elif b in obs:  # a 缺测
                    k = miss.index(a)
                    moment = cond_mean[k] * vals[b]
                else:  # 两者都缺测
                    ka, kb = miss.index(a), miss.index(b)
                    moment = cond_mean[ka] * cond_mean[kb] + cond_cov[ka][kb]
                st['s2'][a][b] += moment
    return stats, imputations


def _merge_stats(target, part):
    for c, st in part.items():
        t = target[c]
        t['n'] += st['n']
        for a in range(len(t['s1'])):
            t['s1'][a] += st['s1'][a]
            for b in range(len(t['s1'])):
                t['s2'][a][b] += st['s2'][a][b]


def _m_step(agg, classes, smoothing):
    d = len(FEATURES)
    means, covs, priors, ridges = {}, {}, {}, {}
    total = sum(agg[c]['n'] for c in classes)
    for c in classes:
        st = agg[c]
        n = st['n']
        if n <= 0:
            raise AnalysisError('某一故障类型在训练组中没有样本，无法估计参数')
        mean = [st['s1'][a] / n for a in range(d)]
        cov = mat_zeros(d, d)
        for a in range(d):
            for b in range(a, d):
                v = st['s2'][a][b] / n - mean[a] * mean[b]
                cov[a][b] = cov[b][a] = v
        cov, ridge = _regularize(cov, smoothing)
        # 提前暴露数值问题
        cholesky(cov)
        means[c] = mean
        covs[c] = cov
        priors[c] = n / total
        ridges[c] = ridge
    return means, covs, priors, ridges


def _em_fold(train_idx, classes, class_of, smoothing, batch_size, checkpoint, fold_no=1):
    """单折训练组上的分批 EM。检查点按轮恢复，重试不重复计入。"""
    d = len(FEATURES)
    class_idx = [[] for _ in classes]
    for i in train_idx:
        class_idx[class_of[i]].append(i)
    for ci, c in enumerate(classes):
        if len(class_idx[ci]) == 0:
            raise AnalysisError('故障类型「%s」在某折训练组中没有样本，请增加样本或减少折数' % c)
    # 每折训练组独立做同测缺口预检，全局通过但折内不满足时同样停算并指明缺口
    _check_co_measurement(labels_global, observed_global, classes, subset=train_idx, fold=fold_no)
    means = _initial_means(values_global, observed_global, classes, class_idx)
    covs = _initial_cov(values_global, observed_global, classes, class_idx, means)
    covs = {c: _regularize(covs[c], smoothing)[0] for c in classes}

    batches = [train_idx[i:i + batch_size] for i in range(0, len(train_idx), batch_size)]
    start_iter = 0
    if checkpoint is not None and checkpoint.get('iteration'):
        start_iter = checkpoint['iteration']
        means = {c: list(m) for c, m in checkpoint['means'].items()}
        covs = {c: [row[:] for row in cov] for c, cov in checkpoint['covs'].items()}

    history = checkpoint['history'] if checkpoint else []
    last_imputed = 0
    for it in range(start_iter, EM_MAX_ITERS):
        params = (means, covs)  # 本轮各批共用同一套上轮参数
        agg = {c: {'n': 0, 's1': [0.0] * d, 's2': mat_zeros(d, d)} for c in classes}
        imputed = 0
        for bi, batch in enumerate(batches):
            part, n_imp = _accumulate_batch(batch, params, classes, class_of)
            _merge_stats(agg, part)
            imputed += n_imp
            # 批事件仅用于进度观察；检查点只在整轮完成后落盘，批内中断则该轮整体重算
            yield {'phase': 'batch', 'fold_iteration': it, 'batch': bi + 1,
                   'batches': len(batches), 'imputed': imputed}
        new_means, new_covs, priors, ridges = _m_step(agg, classes, smoothing)
        shift = max(max(abs(new_means[c][a] - means[c][a]) for a in range(d)) for c in classes)
        means, covs = new_means, new_covs
        last_imputed = imputed
        history.append({'iteration': it + 1, 'mean_shift': shift,
                        'imputed_cells': imputed, 'batches': len(batches)})
        # 轮次检查点：只有整轮全部批次完成才写入，恢复从最近完成轮开始
        yield {'phase': 'iteration_done', 'iteration': it + 1, 'mean_shift': shift,
               'means': means, 'covs': covs, 'history': history}
        if shift < EM_TOL:
            break
    priors = {c: len(class_idx[ci]) / len(train_idx) for ci, c in enumerate(classes)}
    return {'means': means, 'covs': covs, 'priors': priors,
            'iterations': len(history), 'history': history,
            'imputed_cells': last_imputed, 'batches': len(batches)}


def _posterior_observed(i, model, classes):
    """验证件只用实测维度计算各类后验，缺项完全不进入似然。"""
    obs = list(observed_global[i])
    x = [values_global[i][j] for j in obs]
    logs = []
    for c in classes:
        mean = [model['means'][c][j] for j in obs]
        cov = [[model['covs'][c][a][b] for b in obs] for a in obs]
        logs.append(math.log(model['priors'][c]) + gaussian_logpdf(x, mean, cov))
    m = max(logs)
    exps = [math.exp(v - m) if v > EPS_LOG else 0.0 for v in logs]
    z = sum(exps) or 1.0
    return [v / z for v in exps]


# 分批 E 步通过生成器访问全局解析数据，避免每批复制整表
values_global = None
observed_global = None
labels_global = None


def calculate(records, parameters, checkpoint=None, progress=None):
    """主入口。checkpoint 为可选的恢复检查点字典；progress 为可选回调。"""
    global values_global, observed_global, labels_global
    p = validate_parameters(parameters)
    refs, labels, observed, values = _parse_records(records)
    values_global, observed_global, labels_global = values, observed, labels
    n = len(labels)
    classes = sorted(set(labels))
    counts = {c: labels.count(c) for c in classes}
    if len(classes) < 2 or min(counts.values()) < p['folds']:
        raise AnalysisError('至少两种故障类型，且每类样本数不少于交叉验证折数')
    if n < 12:
        raise AnalysisError('分类研究至少需要12条样本')
    _check_co_measurement(labels, observed, classes)

    folds_plan = _stratified_folds(labels, p['folds'])
    class_of = [classes.index(lab) for lab in labels]
    fold_results = []
    pred_by_index = {}
    prob_by_index = {}
    missing_total = sum(1 for obs in observed if len(obs) < len(FEATURES))
    for fold, (train, test) in enumerate(folds_plan, 1):
        cp = None
        if checkpoint and checkpoint.get('fold') == fold:
            cp = checkpoint
        gen = _em_fold(train, classes, class_of, p['smoothing'],
                       DEFAULT_BATCH_SIZE, cp, fold_no=fold)
        model = None
        while True:
            try:
                ev = next(gen)
            except StopIteration as stop:
                model = stop.value
                break
            if progress:
                progress({'fold': fold, **ev})
        correct = 0
        for i in test:
            pr = _posterior_observed(i, model, classes)
            best = max(range(len(classes)), key=lambda k: pr[k])
            pred_by_index[i] = classes[best]
            prob_by_index[i] = pr
            if classes[best] == labels[i]:
                correct += 1
        fold_results.append({
            'fold': fold,
            'training': len(train),
            'validation': len(test),
            'accuracy': correct / len(test),
            'training_indices': train,
            'validation_indices': test,
            'em_iterations': model['iterations'],
            'em_history': model['history'],
            'imputed_cells': model['imputed_cells'],
            'batches': model['batches'],
            'class_priors': {c: model['priors'][c] for c in classes},
            'class_means': {c: model['means'][c] for c in classes},
        })

    # 混淆矩阵
    matrix = [[0] * len(classes) for _ in classes]
    pos = {c: k for k, c in enumerate(classes)}
    correct_all = 0
    for i in range(n):
        matrix[pos[labels[i]]][pos[pred_by_index[i]]] += 1
        if pred_by_index[i] == labels[i]:
            correct_all += 1
    accuracy = correct_all / n
    f1_macro = _macro_f1(matrix, classes)

    rows = []
    missing_readings = []
    for i in range(n):
        pr = prob_by_index[i]
        best = max(range(len(classes)), key=lambda k: pr[k])
        missing_features = [FEATURES[j] for j in range(len(FEATURES)) if j not in observed[i]]
        row = {
            'reference': refs[i],
            'actual': labels[i],
            'predicted': pred_by_index[i],
            'posterior': pr[best],
            'observed_count': len(observed[i]),
            'missing_features': '、'.join(missing_features) if missing_features else '',
            'is_missing_record': bool(missing_features),
        }
        rows.append(row)
        if missing_features:
            # 各折齐备后才汇总缺测件的折外判读
            missing_readings.append({
                'reference': refs[i],
                'actual': labels[i],
                'predicted': pred_by_index[i],
                'posterior': pr[best],
                'observed_features': '、'.join(FEATURES[j] for j in observed[i]),
                'missing_features': row['missing_features'],
                'posterior_by_class': {c: pr[k] for k, c in enumerate(classes)},
            })

    all_folds_complete = len(fold_results) == p['folds']
    result = {
        'summary': [
            {'label': '交叉验证准确率', 'value': accuracy},
            {'label': '宏平均 F1', 'value': f1_macro},
            {'label': '样本数', 'value': n},
            {'label': '类型数', 'value': len(classes)},
            {'label': '缺测件数', 'value': missing_total},
        ],
        'labels': classes,
        'confusion_matrix': matrix,
        'folds': fold_results,
        'rows': rows,
        'columns': ['reference', 'actual', 'predicted', 'posterior', 'observed_count', 'missing_features'],
        'chart': [{'label': c, 'value': counts[c]} for c in classes],
        'features': FEATURES,
        'missing_data': {
            'supported': True,
            'missing_records': missing_total,
            'all_folds_complete': all_folds_complete,
            'co_measurement_check': 'passed',
            'readings': missing_readings if all_folds_complete else [],
            'note': '缺测件带故障标签与至少一项实测值入批；训练缺项按类条件高斯 EM 迭代补全，'
                    '验证件仅用实测维度计算后验；缺项在明细中保留，不做静默填充。',
        },
        'explanation': '全部预测均为折外预测；缺测件按实测维度参与，EM 补全只作用于训练组缺项；'
                       '模型后验未做校准，不能当作检修正确率保证。',
        'checksum': hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'seed': 42,
        'engine': 'stdlib-gaussian-em',
    }
    return result


def _macro_f1(matrix, classes):
    f1s = []
    for k in range(len(classes)):
        tp = matrix[k][k]
        fp = sum(matrix[i][k] for i in range(len(classes))) - tp
        fn = sum(matrix[k]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)
