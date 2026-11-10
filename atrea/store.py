"""Two-layer experiment memory backed by SQLite; raw facts are immutable."""
from __future__ import annotations
import sqlite3, json, uuid
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS experiments(
 experiment_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, created_at TEXT NOT NULL,
 dataset TEXT NOT NULL, split_hash TEXT NOT NULL, model_name TEXT NOT NULL,
 model_version TEXT, config_json TEXT NOT NULL, hyperparameters_json TEXT NOT NULL,
 metrics_json TEXT NOT NULL, seed INTEGER, code_version TEXT, data_hash TEXT,
 paper_ids_json TEXT, hypothesis_id TEXT, status TEXT NOT NULL DEFAULT 'completed',
 memory_scope TEXT NOT NULL DEFAULT 'default');
CREATE TABLE IF NOT EXISTS patterns(
 pattern_id TEXT PRIMARY KEY, statement TEXT NOT NULL, scope_json TEXT NOT NULL,
 support_count INTEGER NOT NULL, effect_json TEXT NOT NULL,
 evidence_experiment_ids_json TEXT NOT NULL, confidence REAL NOT NULL,
 status TEXT NOT NULL DEFAULT 'candidate', created_at TEXT NOT NULL,
 memory_scope TEXT NOT NULL DEFAULT 'default');
CREATE TABLE IF NOT EXISTS abstracts(
 abstract_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL UNIQUE,
 title TEXT NOT NULL, abstract TEXT NOT NULL, methods TEXT NOT NULL,
 results TEXT NOT NULL, discussion TEXT NOT NULL, conclusion TEXT NOT NULL,
 experiment_ids_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS papers(
 paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
 published TEXT, abstract TEXT, source TEXT, relevance REAL,
 extracted_methods_json TEXT, adoption_status TEXT DEFAULT 'unreviewed',
 discovered_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hypotheses(
 hypothesis_id TEXT PRIMARY KEY, statement TEXT NOT NULL, rationale TEXT,
 paper_ids_json TEXT, pattern_ids_json TEXT, experiment_plan_json TEXT,
 status TEXT DEFAULT 'proposed', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_exp_batch ON experiments(batch_id);
CREATE INDEX IF NOT EXISTS idx_exp_model ON experiments(model_name);
"""

class ExperienceStore:
    def __init__(self, path="agent_workspace/atrea_experience.sqlite3"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.session() as c:
            c.executescript(SCHEMA)
            # Backward-compatible migration for databases created by earlier ATREA builds.
            columns = {row[1] for row in c.execute("PRAGMA table_info(experiments)")}
            if "memory_scope" not in columns:
                c.execute("ALTER TABLE experiments ADD COLUMN memory_scope TEXT NOT NULL DEFAULT 'default'")
            pattern_columns = {row[1] for row in c.execute("PRAGMA table_info(patterns)")}
            if "memory_scope" not in pattern_columns:
                c.execute("ALTER TABLE patterns ADD COLUMN memory_scope TEXT NOT NULL DEFAULT 'default'")
    def connect(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        return c
    @contextmanager
    def session(self):
        """Connection context that both commits on success and closes the handle.

        sqlite3's own context manager only commits/rolls back transactions and
        never closes the connection, which keeps the database file locked
        (e.g. preventing temporary-directory cleanup on Windows).
        """
        c = self.connect()
        try:
            with c:
                yield c
        finally:
            c.close()
    @staticmethod
    def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
    def add_experiment(self, record):
        required=("batch_id","dataset","split_hash","model_name","config","hyperparameters","metrics")
        missing=[k for k in required if k not in record]
        if missing: raise ValueError(f"Missing experiment fields: {missing}")
        eid=record.get("experiment_id") or str(uuid.uuid4())
        with self.session() as c:
            c.execute("""INSERT INTO experiments
                (experiment_id,batch_id,created_at,dataset,split_hash,model_name,model_version,
                 config_json,hyperparameters_json,metrics_json,seed,code_version,data_hash,
                 paper_ids_json,hypothesis_id,status,memory_scope)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (eid,record["batch_id"],record.get("created_at",self.now()),record["dataset"],
             record["split_hash"],record["model_name"],record.get("model_version"),
             json.dumps(record["config"],ensure_ascii=False,sort_keys=True),
             json.dumps(record["hyperparameters"],ensure_ascii=False,sort_keys=True),
             json.dumps(record["metrics"],ensure_ascii=False,sort_keys=True),
             record.get("seed"),record.get("code_version"),record.get("data_hash"),
             json.dumps(record.get("paper_ids",[])),record.get("hypothesis_id"),
             record.get("status","completed"),record.get("memory_scope","default")))
        return eid
    def add_pattern(self, statement, scope, support_ids, effect, confidence, status="candidate", memory_scope="default"):
        pid=str(uuid.uuid4())
        with self.session() as c:
            c.execute("""INSERT INTO patterns
                (pattern_id,statement,scope_json,support_count,effect_json,
                 evidence_experiment_ids_json,confidence,status,created_at,memory_scope)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (pid,statement,json.dumps(scope,ensure_ascii=False),len(support_ids),
                 json.dumps(effect,ensure_ascii=False),json.dumps(support_ids),
                 float(confidence),status,self.now(),memory_scope))
        return pid
    def add_paper(self, paper):
        pid=paper.get("paper_id") or str(uuid.uuid4())
        with self.session() as c:
            c.execute("""INSERT OR IGNORE INTO papers VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (pid,paper["title"],paper["url"],paper.get("published"),paper.get("abstract",""),
                 paper.get("source","unknown"),paper.get("relevance"),
                 json.dumps(paper.get("extracted_methods",[]),ensure_ascii=False),
                 paper.get("adoption_status","unreviewed"),self.now()))
        return pid
    def add_hypothesis(self, statement, rationale, paper_ids, pattern_ids, plan):
        hid=str(uuid.uuid4())
        with self.session() as c:
            c.execute("INSERT INTO hypotheses VALUES(?,?,?,?,?,?,?,?)",
              (hid,statement,rationale,json.dumps(paper_ids),json.dumps(pattern_ids),
               json.dumps(plan,ensure_ascii=False),"proposed",self.now()))
        return hid
    def add_abstract(self, batch_id, title, sections, experiment_ids):
        aid=str(uuid.uuid4())
        with self.session() as c:
            c.execute("INSERT INTO abstracts VALUES(?,?,?,?,?,?,?,?,?,?)",
              (aid,batch_id,title,sections["abstract"],sections["methods"],
               sections["results"],sections["discussion"],sections["conclusion"],
               json.dumps(experiment_ids),self.now()))
        return aid
    def batch_experiments(self,batch_id):
        with self.session() as c:
            rows=c.execute("SELECT experiment_id,dataset,model_name,metrics_json,config_json FROM experiments WHERE batch_id=?",(batch_id,)).fetchall()
        return [{"experiment_id":r[0],"dataset":r[1],"model_name":r[2],
                 "metrics":json.loads(r[3]),"config":json.loads(r[4])} for r in rows]

    def latest_paper_date(self):
        with self.session() as c:
            row=c.execute("SELECT MAX(substr(published,1,10)) FROM papers WHERE published IS NOT NULL").fetchone()
        return row[0] if row else None

    def retrieve_experience(self, query=None, *, limit=20, dataset=None, memory_scope="default"):
        """Return prior experiment facts for one isolated memory scope."""
        with self.session() as c:
            sql = "SELECT experiment_id,batch_id,dataset,split_hash,model_name,model_version,config_json,hyperparameters_json,metrics_json,status FROM experiments WHERE memory_scope=?"
            args=[memory_scope]
            if dataset:
                sql += " AND dataset=?"; args.append(dataset)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(int(limit))
            rows=c.execute(sql,args).fetchall()
            patterns=c.execute("SELECT pattern_id,statement,scope_json,support_count,effect_json,confidence,status FROM patterns WHERE memory_scope=? ORDER BY created_at DESC LIMIT ?",(memory_scope,int(limit))).fetchall()
        experiments=[{"experiment_id":r[0],"batch_id":r[1],"dataset":r[2],"split_hash":r[3],
            "model_name":r[4],"model_version":r[5],"config":json.loads(r[6]),
            "hyperparameters":json.loads(r[7]),"metrics":json.loads(r[8]),"status":r[9]} for r in rows]
        if query:
            terms=[x.lower() for x in str(query).split() if len(x)>2]
            if terms:
                def exp_score(e):
                    hay=(e["model_name"]+" "+e["dataset"]+" "+json.dumps(e["config"])).lower()
                    return sum(t in hay for t in terms)
                def pat_score(p):
                    hay=(p[1]+" "+p[2]).lower()
                    return sum(t in hay for t in terms)
                # Rank relevant facts first but retain recent fallback evidence when no
                # exact lexical match exists; never silently erase the entire memory.
                experiments.sort(key=lambda e: exp_score(e), reverse=True)
                patterns.sort(key=lambda p: pat_score(p), reverse=True)
        return {"experiments":experiments,"patterns":[{"pattern_id":r[0],"statement":r[1],
            "scope":json.loads(r[2]),"support_count":r[3],"effect":json.loads(r[4]),
            "confidence":r[5],"status":r[6]} for r in patterns]}

    def recent_papers(self, limit=20):
        with self.session() as c:
            rows=c.execute("SELECT paper_id,title,url,published,abstract,source,extracted_methods_json,adoption_status FROM papers ORDER BY discovered_at DESC LIMIT ?",(int(limit),)).fetchall()
        return [{"paper_id":r[0],"title":r[1],"url":r[2],"published":r[3],"abstract":r[4],
                 "source":r[5],"extracted_methods":json.loads(r[6]),"adoption_status":r[7]} for r in rows]

    def induce_pairwise_patterns(self, *, min_support=3, relative_effect=0.03, memory_scope="default"):
        """Induce conservative paired-seed patterns on identical dataset/split scopes.

        Only candidate evidence is emitted; a pattern is not treated as causal truth.
        """
        with self.session() as c:
            rows=c.execute("SELECT experiment_id,dataset,split_hash,model_name,metrics_json,seed FROM experiments WHERE status='completed' AND memory_scope=?",(memory_scope,)).fetchall()
        groups={}
        for eid,dataset,split,model,metrics_json,seed in rows:
            metrics=json.loads(metrics_json)
            if isinstance(metrics.get('mae'),(int,float)) and seed is not None:
                groups.setdefault((dataset,split),[]).append((eid,model,int(seed),float(metrics['mae'])))
        created=[]
        for (dataset,split),facts in groups.items():
            by_model={}
            for eid,model,seed,mae in facts:
                by_model.setdefault(model,{})[seed]=(eid,mae)
            models=sorted(by_model)
            for i,model_a in enumerate(models):
                for model_b in models[i+1:]:
                    common=sorted(set(by_model[model_a]) & set(by_model[model_b]))
                    if len(common)<min_support: continue
                    a=np_mean([by_model[model_a][seed][1] for seed in common])
                    b=np_mean([by_model[model_b][seed][1] for seed in common])
                    if min(a,b)<=0: continue
                    better,worse=(model_a,model_b) if a<=b else (model_b,model_a)
                    better_mae=min(a,b); worse_mae=max(a,b)
                    effect=(worse_mae-better_mae)/worse_mae
                    if effect<relative_effect: continue
                    ids=[by_model[better][seed][0] for seed in common]+[by_model[worse][seed][0] for seed in common]
                    statement=(f"{better} has lower paired-seed MAE than {worse} on {dataset} "
                               f"(same split; {len(common)} common seeds; relative difference {effect:.1%}).")
                    pid=self.add_pattern(statement,{"dataset":dataset,"split_hash":split,"paired_seeds":common},ids,
                        {"metric":"mae","relative_difference":effect,"paired_seed_count":len(common)},
                        confidence=min(0.99,0.5+0.1*len(common)),memory_scope=memory_scope)
                    created.append(pid)
        return created


def np_mean(values):
    return sum(values) / len(values) if values else 0.0
