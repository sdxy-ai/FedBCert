import pytest
import torch
from scripts import fedbcert_rg1_evaluation as ev
from scripts.fedbcert_rg1_training import make_model

def fixture():
    torch.manual_seed(42020000); spec=dict(name="P",input_dim=3,num_classes=3,hidden_dim=5)
    models={name:make_model(dict(spec,name=name)).double().eval() for name in ("P","I","MLP")}
    models["I"].load_state_dict(models["P"].state_dict())
    allowed=dict(features=torch.randn(8,3,dtype=torch.float64),full_edge_index=torch.tensor([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]]),membership=torch.tensor([0,0,1,1,2,2,0,2]))
    return models,allowed

def test_local_never_constructs_remote_service(monkeypatch):
    models,allowed=fixture()
    def prohibited(*args,**kwargs): raise AssertionError("remote service")
    monkeypatch.setattr(ev.legacy,"AggregateReplayService",prohibited)
    assert torch.isfinite(ev._local(models["P"],ev._observed(allowed),torch.arange(8))).all()
    with pytest.raises(RuntimeError,match="remote topology"): ev._local(models["P"],allowed,torch.arange(8))

def test_identity_controls_and_fingerprint():
    models,allowed=fixture(); obs=ev._observed(allowed); other=dict(obs,local_edge_index=torch.empty((2,0),dtype=torch.long))
    for name in ("I","MLP"):
        assert torch.equal(ev._local(models[name],obs,torch.arange(8),True),ev._local(models[name],other,torch.arange(8),True))
    before=ev._fingerprint(models["MLP"]); models["MLP"].encoder_norms[0].eps=.1
    assert before!=ev._fingerprint(models["MLP"])

def test_all_policies_and_exclusive_evidence(tmp_path):
    models,allowed=fixture(); models["P"].theta.data.copy_(torch.tensor([.5,-.3,.2]))
    result=ev.evaluate_case(models,allowed,torch.arange(8),torch.arange(8)%3,tmp_path/"case",0,42020000)
    rows=result["timing_records"]; assert len(rows)==64*1
    assert result["structural_diagnostics"]["all_graph"]["n"]==8
    assert result["structural_diagnostics"]["all_test"]["n"]==8
    assert set(result["structural_diagnostics"]["exposed_test"]["coverage"])=={"S","D","N"}
    measured=[r for r in rows if not r["warmup"]]; assert len(measured)==48
    assert all(r["total_wire_bytes"]==0 for r in measured if r["method"] in {"L","I","MLP"})
    assert all(r["class_mismatches"]==0 for r in measured if r["method"] in {"F","E","S","D","N"})
    with pytest.raises(FileExistsError): ev._publish(tmp_path/"case/summary.json",{})

def test_label_payload_rejected_before_policy(tmp_path):
    models,allowed=fixture()
    with pytest.raises(RuntimeError,match="label-free"): ev.evaluate_case(models,dict(allowed,labels=torch.zeros(8)),torch.arange(8),torch.zeros(8),tmp_path,0,1)

def test_forced_fallback_class_flip_is_hard_error(tmp_path,monkeypatch):
    models,allowed=fixture(); original=ev.legacy.run_strategy
    def corrupt(*args):
        row,evidence=original(*args)
        if args[3]=="full_reference":
            changed=evidence["final_logits"].clone(); alternate=(changed.argmax(1)+1)%changed.size(1)
            changed.zero_(); changed.scatter_(1,alternate[:,None],100.); evidence["final_logits"]=changed
        return row,evidence
    monkeypatch.setattr(ev.legacy,"run_strategy",corrupt)
    with pytest.raises(RuntimeError,match="class/numerical replay mismatch"): ev.evaluate_case(models,allowed,torch.arange(8),torch.arange(8)%3,tmp_path,0,1)
    assert not (tmp_path/"summary.json").exists()
