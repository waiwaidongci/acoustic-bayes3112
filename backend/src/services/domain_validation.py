import math
from ..config.settings import CONFIG
def _is_missing(v):
    return v is None or (isinstance(v,str) and not v.strip())
def validate_record(values):
    out={}
    measured=0
    allow_missing=CONFIG['algorithm']=='gaussian'
    for key,label,kind in CONFIG['fields']:
        v=values.get(key)
        if kind=='number':
            # 缺测件：仅声检（gaussian）域允许特征空值，以 None 保留缺项，不做插补
            if allow_missing and _is_missing(v):
                out[key]=None
                continue
            if isinstance(v,bool):raise ValueError(label+'必须为数值')
            try:v=float(v)
            except (TypeError,ValueError):raise ValueError(label+'必须为数值')
            if not math.isfinite(v):raise ValueError(label+'必须是有限数值')
            out[key]=v;measured+=1
        else:
            if not isinstance(v,str) or not v.strip() or len(v)>600:raise ValueError(label+'不能为空且最多600字符')
            v=v.strip()
        out[key]=v
    kind=CONFIG['algorithm']
    if kind=='gaussian':
        if out.get('label') and measured==0:raise ValueError('缺测件至少保留一项实测声学特征，不能全部缺测')
        if any(out[f] is not None and out[f]<0 for f in ('rms','centroid','kurtosis')):raise ValueError('声学特征不可为负')
    if kind=='density' and (not 0<=out['humidity']<=100 or out['particles']<0):raise ValueError('湿度需在0–100且颗粒浓度非负')
    if kind=='transition':
        parts=[x.strip() for x in out['path'].split('>')]
        if any(not x for x in parts) or len(parts)>80:raise ValueError('章节路径需1–80个非空章节，用>分隔')
        out['path']='>'.join(parts)
    if kind=='forest' and (not 0<out['moisture']<=100 or min(out['thickness'],out['temperature'],out['airflow'],out['minutes'])<=0):raise ValueError('湿度需在(0,100]，其余工艺量为正')
    if kind=='neighbors' and not 1<=out['rating']<=5:raise ValueError('评分需在1–5之间')
    return out
def validate_parameters(params):
    defaults=CONFIG['default_parameters'];out={}
    if set(params)-set(defaults):raise ValueError('包含不支持的分析参数')
    for k,v in defaults.items():
        try:n=float(params.get(k,v))
        except (TypeError,ValueError):raise ValueError('参数必须为有限数值：'+k)
        if not math.isfinite(n):raise ValueError('参数必须为有限数值：'+k)
        out[k]=n
    ranges={'folds':(2,8),'smoothing':(1e-12,.01),'eps':(.01,5),'min_samples':(2,100),'min_count':(1,1000),'top_n':(1,100),'trees':(20,200),'depth':(2,16),'neighbors':(1,20),'min_common':(2,10)}
    for k,n in out.items():
        lo,hi=ranges[k]
        if not lo<=n<=hi:raise ValueError(f'{k}需在{lo}至{hi}之间')
        if k not in ('smoothing','eps'):
            if n!=int(n):raise ValueError(k+'必须为整数')
            out[k]=int(n)
    return out
