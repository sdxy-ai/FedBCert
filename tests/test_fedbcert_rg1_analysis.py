import pytest
import numpy as np
from scripts.fedbcert_rg1_analysis import PAIRS,UTILITY,WORKLOADS,analyze,paired_bootstrap
def toy():
    records=[]; cases=[]; wire=dict(L=0,F=100,E=80,S=70,D=60,N=40); cert=dict(L=0,F=0,E=2,S=4,D=6,N=8)
    for ds in ("Cora","CiteSeer"):
        for part in ("metis","random"):
            for seed in (101,102):
                key=dict(dataset=ds,partition=part,seed=seed)
                cases.append(dict(**key,utility={m:{f:.7-delta for f in UTILITY} for m,delta in (("P",0),("I",.1),("MLP",.05))},coverage=dict(S=.4,D=.6,N=.8)))
                for w in WORKLOADS:
                    for m,s in PAIRS:
                        for rep,cpu in enumerate((1.,50.,3.)): records.append(dict(**key,workload=w,model=m,strategy=s,repetition=rep,cpu_seconds=cpu,total_wire_bytes=wire[s] if m=="P" else 0,certified_n=cert[s] if m=="P" else 10,target_n=10))
    return records,cases
def test_pairing_medians_and_effects():
    records,cases=toy(); result=analyze(records,cases,formal=False,draws=64)
    assert result["record_count"]==384 and result["P_records"]==288 and all(r["cpu_seconds"]==3 for r in result["measurements"])
    assert result["overall"]["utility/P-I/macro_f1"]["estimate"]==pytest.approx(.1)
    assert result["overall"]["all_test/N-vs-F/payload_reduction"]["estimate"]==pytest.approx(.6)
def test_independent_dataset_draws():
    series={"a":[0.,1.],"b":[0.,10.]}; r=paired_bootstrap(series,draws=100,rng_seed=17); rng=np.random.default_rng(17)
    expected=sum(np.array(v)[rng.integers(0,2,(100,2))].mean(1) for v in series.values())/2
    assert r["ci95"]==pytest.approx(np.quantile(expected,[.025,.975]))
def test_zero_denominator_kept():
    records,cases=toy()
    for r in records:
        if (r["dataset"],r["partition"],r["seed"],r["workload"],r["model"],r["strategy"])==("Cora","metis",101,"all_test","P","F"): r["total_wire_bytes"]=0; r["cpu_seconds"]=0.
    result=analyze(records,cases,formal=False,draws=64); value=result["overall"]["all_test/N-vs-F/payload_reduction"]
    assert value["na"]==1 and value["estimate"] is None and value["ci95"] is None
@pytest.mark.parametrize("fault",["missing","duplicate","bytes","coverage","formal"])
def test_invalid_matrix(fault):
    records,cases=toy()
    if fault=="missing": records.pop()
    elif fault=="duplicate": records.append(dict(records[0]))
    elif fault=="bytes": records[0]["total_wire_bytes"]+=1
    elif fault=="coverage": cases[0]["coverage"]["N"]=.9
    with pytest.raises(ValueError): analyze(records,cases,formal=fault=="formal",draws=8)
