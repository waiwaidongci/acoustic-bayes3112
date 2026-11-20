import json,time,logging
from pathlib import Path
from redis import Redis
from sqlalchemy import select
from .config.database import Session
from .config.settings import REDIS_URL,ARTIFACT_DIR
from .models.run import Run
from .models import source,record,user,note,audit
from .services.algorithm import calculate
from .services.audit import log
from .services.export import csv_response
queue=Redis.from_url(REDIS_URL)
ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)

def checkpoint_path(run_id):
    return ARTIFACT_DIR/f'run-{run_id}.checkpoint.json'

def load_checkpoint(run_id):
    """中断恢复：只认最近一次“整轮完成”后落盘的检查点。"""
    p=checkpoint_path(run_id)
    if not p.exists():return None
    try:
        return json.loads(p.read_text())
    except Exception:
        logging.exception('checkpoint unreadable, restart from scratch')
        return None

def save_checkpoint(run_id,event):
    # 仅在整轮全部批次完成后写盘，批内中断则本轮重算，重试不会重复计入
    if event.get('phase')!='iteration_done':return
    cp={'fold':event['fold'],'iteration':event['iteration'],
        'mean_shift':event['mean_shift'],'means':event['means'],
        'covs':event['covs'],'history':event['history'],'updated_at':time.time()}
    tmp=checkpoint_path(run_id).with_suffix('.tmp')
    tmp.write_text(json.dumps(cp,ensure_ascii=False,allow_nan=False))
    tmp.replace(checkpoint_path(run_id))

def execute(id):
    with Session() as db:
        r=db.get(Run,id)
        if not r or r.state!='queued':return
        r.state='running';db.commit()
        checkpoint=load_checkpoint(id)
        try:
            def progress(event):
                save_checkpoint(id,event)
            result=calculate(r.snapshot['records'],r.parameters,checkpoint=checkpoint,progress=progress)
            json.dumps(result,allow_nan=False)
            r.result=result;r.state='completed'
            response=csv_response(result,'run.csv')
            (ARTIFACT_DIR/f'run-{r.id}.csv').write_bytes(response.body)
            # 成功后清理检查点；清理失败不影响已完成结果，失败时检查点保留以便续算
            try:
                cp=checkpoint_path(r.id)
                if cp.exists():cp.unlink()
            except OSError:
                logging.exception('checkpoint cleanup failed for run %s',r.id)
            log(db,'worker','run.completed',r.id)
        except Exception as e:
            logging.exception('analysis failed');r.state='failed';r.error=str(e)[:2000]
            log(db,'worker','run.failed',r.id,r.error)
        db.commit()
def main():
    while True:
        (ARTIFACT_DIR/'heartbeat').write_text(str(time.time()))
        value=queue.blpop('analysis:queue',timeout=2)
        if value:execute(int(value[1]))
if __name__=='__main__':main()
