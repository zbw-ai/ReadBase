import json, subprocess, time, threading, hashlib, csv, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parent
CACHE=Path('/tmp/readbase-history-recheck/cache');CACHE.mkdir(parents=True,exist_ok=True)
START='2024-12-31T16:00:00Z';END='2026-06-30T15:59:59Z'
REPOS=['areal-project/AReaL','verl-project/verl','THUDM/slime','alibaba/ROLL','OpenRLHF/OpenRLHF','NVIDIA-NeMo/RL','NVIDIA/Megatron-LM','vllm-project/vllm','sgl-project/sglang','huggingface/trl','huggingface/transformers','huggingface/accelerate','huggingface/peft','huggingface/kernels','huggingface/tokenizers']
FOUND=['NVIDIA/TransformerEngine','NVIDIA/nccl','Dao-AILab/flash-attention','deepspeedai/DeepSpeed','pytorch/pytorch','deepseek-ai/DeepEP','deepseek-ai/FlashMLA','deepseek-ai/DeepGEMM']
sem=threading.Semaphore(20)
def now():return datetime.now(timezone.utc).isoformat()
def api(endpoint):
 p=CACHE/(hashlib.sha256(endpoint.encode()).hexdigest()+'.json')
 if p.exists():return json.loads(p.read_text())
 for attempt in range(4):
  with sem:r=subprocess.run(['gh','api',endpoint],capture_output=True,text=True,timeout=120)
  if r.returncode==0:
   d=json.loads(r.stdout);p.write_text(json.dumps(d));return d
  time.sleep(2**attempt)
 raise RuntimeError(endpoint+': '+r.stderr[:300])
def pages(repo,kind,suffix=''):
 allitems=[];page=1;finished=False
 with ThreadPoolExecutor(max_workers=4) as pool:
  while not finished:
   nums=range(page,page+4)
   results=list(pool.map(lambda n:api(f'repos/{repo}/{kind}?per_page=100&page={n}'+suffix),nums))
   for n,data in zip(nums,results):
    if not isinstance(data,list):raise ValueError(str(data)[:200])
    allitems.extend(data);page=n+1
    if len(data)<100 or (kind=='pulls' and data and min(x['updated_at'] for x in data)<START):finished=True;break
   if kind=='pulls' and page%40==1:print('PROGRESS',repo,kind,page,flush=True)
 key='sha' if kind=='commits' else 'id'
 unique={x[key]:x for x in allitems}
 return list(unique.values()), {'pages':page-1,'enumerated_unique':len(unique),'complete':True}
def collect(repo,foundation=False):
 status_path=CACHE/(repo.replace('/','_')+'-status.json')
 if status_path.exists():return json.loads(status_path.read_text())
 s={'repo':repo,'query_started':now(),'window_start':START,'window_end':END}
 if not foundation:
  meta=api('repos/'+repo);s.update(default_branch=meta['default_branch'],repo_created_at=meta['created_at'])
  s['head_at_query']=api('repos/'+repo+'/branches/'+s['default_branch'])['commit']['sha']
  commits,cm=pages(repo,'commits','&sha='+s['head_at_query']+'&since='+START+'&until='+END)
  commits=[x for x in commits if START<=x['commit']['committer']['date']<=END]
  pulls,pm=pages(repo,'pulls','&state=closed&sort=updated&direction=desc')
  pulls=[x for x in pulls if x.get('merged_at') and START<=x['merged_at']<=END]
  for kind,data,info in [('commits',commits,cm),('pulls',pulls,pm)]:
   (CACHE/(repo.replace('/','_')+'-'+kind+'.json')).write_text(json.dumps(data));s[kind]={**info,'selected':len(data)}
 releases,rm=pages(repo,'releases')
 releases=[x for x in releases if x.get('published_at') and START<=x['published_at']<=END]
 (CACHE/(repo.replace('/','_')+'-releases.json')).write_text(json.dumps(releases));s['releases']={**rm,'selected':len(releases)}
 s['query_finished']=now();status_path.write_text(json.dumps(s,indent=2));print('DONE',repo,{k:s[k]['selected'] for k in ['commits','pulls','releases'] if k in s},flush=True);return s
with ThreadPoolExecutor(max_workers=5) as p:
 statuses=list(p.map(collect,REPOS))
with ThreadPoolExecutor(max_workers=5) as p:
 foundations=list(p.map(lambda r:collect(r,True),FOUND))
def period(dt):
 d=datetime.fromisoformat(dt.replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8)))
 return f'2025-Q{(d.month-1)//3+1}' if d.year==2025 else d.strftime('%Y-%m')
periods=[f'2025-Q{i}' for i in range(1,5)]+[f'2026-{i:02}' for i in range(1,7)]
rows={q:{k:[] for k in ['commits','pulls','releases']} for q in periods}
for s in statuses:
 repo=s['repo']
 for kind in rows[periods[0]]:
  data=json.loads((CACHE/(repo.replace('/','_')+'-'+kind+'.json')).read_text())
  for x in data:
   if kind=='commits':dt=x['commit']['committer']['date'];row=[repo,x['sha'],dt,x['commit']['message'].splitlines()[0],(x.get('author') or {}).get('login',x['commit']['author'].get('name','')),s['default_branch'],x['sha'],x['html_url']]
   elif kind=='pulls':dt=x['merged_at'];row=[repo,str(x['number']),dt,x['title'],x['user']['login'],x['base']['ref'],x['merge_commit_sha'],x['html_url']]
   else:dt=x['published_at'];row=[repo,str(x['id']),dt,x.get('name') or x['tag_name'],x['author']['login'],x['tag_name'],x['target_commitish'],x['html_url']]
   rows[period(dt)][kind].append(row)
manifest={'status':'COMPLETE_EVENT_INDEX_NOT_CODE_AUDIT','historical_review_date':'2026-09-22','index_rechecked_at':now(),'timezone':'Asia/Shanghai','start_utc':START,'end_utc':END,'repositories':statuses,'foundation_repositories':foundations,'files':[],'totals':{},'periods':{},'method':'Pinned default-branch head commits by committer time; closed PRs sorted by update time paginated past lower boundary then filtered by merged_at; release endpoint exhausted and filtered by published_at. Indexed metadata is not full code review. Live API pagination is not an atomic snapshot; later rebases, deleted/private history and other branches may be absent.'}
def write_csv(path,header,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(data)
 manifest['files'].append({'path':str(path.relative_to(ROOT)),'rows':len(data),'columns':len(header),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
header=['repo','source_id','event_time_utc','title','author','branch_or_tag','commit_or_target','url']
for q in periods:
 manifest['periods'][q]={k:len(v) for k,v in rows[q].items()}
 for kind,data in rows[q].items():write_csv(ROOT/'github'/q/(kind+'.csv'),header,sorted(data,key=lambda r:(r[2],r[0],r[1])))
manifest['totals']={k:sum(len(rows[q][k]) for q in periods) for k in ['commits','pulls','releases']}
frows=[]
for s in foundations:
 for x in json.loads((CACHE/(s['repo'].replace('/','_')+'-releases.json')).read_text()):frows.append([s['repo'],str(x['id']),x['published_at'],x.get('name') or x['tag_name'],x['author']['login'],x['tag_name'],x['target_commitish'],x['html_url']])
write_csv(ROOT/'foundation_releases.csv',header,sorted(frows,key=lambda r:(r[2],r[0])))
manifest['foundation_release_count']=len(frows)
(ROOT/'github_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('COMPLETE',manifest['totals'],'foundation',len(frows),flush=True)
