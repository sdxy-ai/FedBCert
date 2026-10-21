"""Dataset-stratified paired descriptive analysis; missing ratios stay missing."""
from itertools import product
from math import isfinite
from statistics import median
import numpy as np
DATASETS=("Cora","CiteSeer"); PARTITIONS=("metis","random"); SEEDS=tuple(range(42020101,42020109)); WORKLOADS=("all_test","small_batch")
PAIRS=tuple(("P",s) for s in ("L","F","E","S","D","N"))+(("I","L"),("MLP","L"))
UTILITY=("macro_f1","accuracy","exposed_macro_f1","worst_client_f1")
def _number(v,nullable=False,unit=False):
    if v is None and nullable: return None
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or (unit and not 0<=v<=1): raise ValueError("invalid metric")
    return float(v)
def _mean(v):
    v=list(v); return None if any(x is None for x in v) else float(np.mean(v))
def _difference(a,b): return None if a is None or b is None else a-b
def _ratio(a,b): return None if b==0 else a/b
def _case(row): return row["dataset"],row["partition"],row["seed"]
def paired_bootstrap(values_by_dataset,*,draws=10000,rng_seed=42028616):
    if not values_by_dataset or draws<1: raise ValueError("empty bootstrap")
    rng=np.random.default_rng(rng_seed); samples=np.zeros(draws); valid=np.ones(draws,dtype=bool); means=[]; n=na=0
    for values in values_by_dataset.values():
        values=[_number(v,nullable=True) for v in values]
        if not values: raise ValueError("empty stratum")
        a=np.asarray([np.nan if v is None else v for v in values]); n+=len(a); na+=sum(v is None for v in values); means.append(_mean(values))
        picked=a[rng.integers(0,len(a),(draws,len(a)))]; valid &= np.isfinite(picked).all(1); samples+=np.where(np.isfinite(picked),picked,0.).mean(1)
    samples/=len(values_by_dataset)
    return dict(estimate=_mean(means),ci95=None if na else np.quantile(samples,[.025,.975]).tolist(),n=n,na=na,bootstrap_na=int((~valid).sum()))
def analyze(records,case_metrics,*,formal=True,draws=10000,rng_seed=42028616):
    records=list(records); case_metrics=list(case_metrics); metrics={_case(r):r for r in case_metrics}
    if not metrics or len(metrics)!=len(case_metrics): raise ValueError("empty/duplicate metrics")
    datasets,partitions,seeds=(DATASETS,PARTITIONS,SEEDS) if formal else tuple(tuple(sorted({k[i] for k in metrics})) for i in range(3))
    cases=tuple(product(datasets,partitions,seeds))
    if set(cases)!=set(metrics): raise ValueError("incomplete case matrix")
    groups={}
    for r in records:
        case=_case(r); pair=(r["model"],r["strategy"]); rep=r["repetition"]
        if case not in metrics or r["workload"] not in WORKLOADS or pair not in PAIRS or type(rep) is not int or rep not in (0,1,2): raise ValueError("unexpected measurement")
        for field in ("total_wire_bytes","certified_n","target_n"):
            if type(r[field]) is not int or r[field]<0: raise ValueError("invalid count")
        if r["certified_n"]>r["target_n"] or _number(r["cpu_seconds"])<0: raise ValueError("count/CPU")
        key=(case,r["workload"],*pair); bucket=groups.setdefault(key,{})
        if rep in bucket: raise ValueError("duplicate measurement")
        bucket[rep]=r
    if len(records)!=len(cases)*2*8*3: raise ValueError("incomplete measurements")
    lookup={}; measurements=[]
    for case,workload,pair in product(cases,WORKLOADS,PAIRS):
        key=(case,workload,*pair); bucket=groups.get(key,{})
        if set(bucket)!={0,1,2}: raise ValueError("incomplete repetitions")
        invariant={(r["total_wire_bytes"],r["certified_n"],r["target_n"]) for r in bucket.values()}
        if len(invariant)!=1: raise ValueError("repeated bytes/counts changed")
        wire,cert,target=invariant.pop(); row=dict(dataset=case[0],partition=case[1],seed=case[2],workload=workload,model=pair[0],strategy=pair[1],cpu_seconds=float(median(r["cpu_seconds"] for r in bucket.values())),total_wire_bytes=wire,certified_n=cert,target_n=target)
        lookup[key]=row; measurements.append(row)
    for case,workload in product(cases,WORKLOADS):
        if len({lookup[(case,workload,*p)]["target_n"] for p in PAIRS})!=1: raise ValueError("different targets")
    values={}
    for case in cases:
        source=metrics[case]; flat={}; utility={m:{f:_number(source["utility"][m][f],nullable=True,unit=True) for f in UTILITY} for m in ("P","I","MLP")}; coverage={s:_number(source["coverage"][s],nullable=True,unit=True) for s in ("S","D","N")}
        for m,f in product(utility,UTILITY): flat[f"utility/{m}/{f}"]=utility[m][f]
        for m,f in product(("I","MLP"),UTILITY): flat[f"utility/P-{m}/{f}"]=_difference(utility["P"][f],utility[m][f])
        for s in coverage:
            r=lookup[(case,"all_test","P",s)]; ratio=_ratio(r["certified_n"],r["target_n"])
            if (ratio is None)!=(coverage[s] is None) or (ratio is not None and abs(ratio-coverage[s])>1e-12): raise ValueError("coverage/count mismatch")
            flat[f"coverage/{s}"]=coverage[s]
        for s in ("S","D"): flat[f"coverage/N-{s}"]=_difference(coverage["N"],coverage[s])
        structural=source.get("structural_diagnostics")
        if formal and structural is None: raise ValueError("structural diagnostics missing")
        if structural is not None:
            for group in ("all_graph","exposed_graph","all_test","exposed_test"):
                row=structural[group]
                for field in ("class_change_rate","probability_change_at_least_005"):
                    flat[f"structural/{group}/{field}"]=_number(row[field],nullable=True,unit=True)
                for s in ("S","D","N"):
                    flat[f"structural/{group}/coverage/{s}"]=_number(row["coverage"][s],nullable=True,unit=True)
        for w in WORKLOADS:
            for m,s in PAIRS:
                r=lookup[(case,w,m,s)]; prefix=f"{w}/{m}/{s}"
                for f in ("cpu_seconds","total_wire_bytes"): flat[f"{prefix}/{f}"]=r[f]
                flat[f"{prefix}/certified_ratio"]=_ratio(r["certified_n"],r["target_n"])
            chosen=lookup[(case,w,"P","N")]
            for s in ("F","D","E"):
                r=lookup[(case,w,"P",s)]; prefix=f"{w}/N-vs-{s}"; ratio=_ratio(chosen["total_wire_bytes"],r["total_wire_bytes"])
                flat[f"{prefix}/payload_reduction"]=None if ratio is None else 1-ratio; flat[f"{prefix}/cpu_ratio"]=_ratio(chosen["cpu_seconds"],r["cpu_seconds"])
                flat[f"{prefix}/absolute_payload_difference"]=chosen["total_wire_bytes"]-r["total_wire_bytes"]
        values[case]=flat
    names=tuple(values[cases[0]])
    # One fixed dataset-specific draw matrix is shared across both partitions,
    # every metric and all table levels; the two datasets have independent draws.
    rng=np.random.default_rng(rng_seed)
    indices={ds:rng.integers(0,len(seeds),(draws,len(seeds))) for ds in datasets}
    def summarize(strata):
        output={}
        for name in names:
            vectors={ds:[r[name] for r in rows] for ds,rows in strata.items()}
            na=sum(v is None for vector in vectors.values() for v in vector); valid=np.ones(draws,dtype=bool); samples=np.zeros(draws)
            for ds,vector in vectors.items():
                a=np.array([np.nan if v is None else v for v in vector]); picked=a[indices[ds]]
                valid &= np.isfinite(picked).all(1); samples+=np.where(np.isfinite(picked),picked,0.).mean(1)
            samples/=len(vectors)
            output[name]=dict(estimate=_mean(_mean(v) for v in vectors.values()),ci95=None if na else np.quantile(samples,[.025,.975]).tolist(),n=sum(map(len,vectors.values())),na=na,bootstrap_na=int((~valid).sum()))
        return output
    partitions_out=[]; datasets_out=[]; paired={}
    for ds in datasets:
        for part in partitions:
            rows=[values[(ds,part,s)] for s in seeds]; partitions_out.append(dict(dataset=ds,partition=part,seed_rows=[dict(seed=s,metrics=r) for s,r in zip(seeds,rows)],summary=summarize({ds:rows})))
        rows=[{name:_mean(values[(ds,p,s)][name] for p in partitions) for name in names} for s in seeds]; paired[ds]=rows
        datasets_out.append(dict(dataset=ds,seed_rows=[dict(seed=s,metrics=r) for s,r in zip(seeds,rows)],summary=summarize({ds:rows})))
    return dict(scope="Descriptive fixed-graph uncertainty, not independent graphs",bootstrap=dict(draws=draws,rng_seed=rng_seed,dataset_order=list(datasets)),case_count=len(cases),record_count=len(records),P_records=sum(r["model"]=="P" for r in records),control_records=sum(r["model"]!="P" for r in records),measurements=measurements,dataset_partition=partitions_out,dataset=datasets_out,overall=summarize(paired),na_counts={name:sum(values[c][name] is None for c in cases) for name in names})
