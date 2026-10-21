"""Engineering-only gates and failure retention; no real-graph predictions."""
from pathlib import Path
import pytest
from scripts import fedbcert_rg1_common as common
from scripts import run_fedbcert_rg1 as runner

@pytest.fixture
def toy(tmp_path): return common.prepare(tmp_path/"inputs",toy=True)

def test_no_vault_before_all_models_and_no_test_capability(toy,tmp_path):
    manifest=common.read(toy); execution=tmp_path/"execution"; execution.mkdir()
    with pytest.raises(FileNotFoundError): common.open_vault(toy,execution,manifest["cases"][0])
    payload=common.load(manifest["matrix"][0]["training_inputs"])
    assert set(payload)=={"clients"}
    for client in payload["clients"]:
        assert set(client)=={"features","local_edge_index","train_targets","validation_targets"}
        assert not ((client["train_targets"]>=0)&(client["validation_targets"]>=0)).any()
    with pytest.raises(FileNotFoundError): common.seal(toy,execution)

def test_input_mutation_and_manifest_change_rejected(toy):
    m=common.read(toy); path=Path(m["matrix"][0]["training_inputs"])
    with path.open("ab") as f: f.write(b"toy-corruption")
    with pytest.raises(RuntimeError,match="frozen file mismatch"): common.validate_manifest(toy)

def test_authorization_bound_to_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"validate_manifest",lambda p:dict(mode="formal_pending_authorization"))
    path=tmp_path/"manifest.json"; common.publish(path,{})
    with pytest.raises(PermissionError): common.authorize(None,path,"train")
    auth=tmp_path/"approval.json"; common.publish(auth,dict(approved=True,manifest_sha256="not-this-file",scopes=["train"]))
    with pytest.raises(PermissionError): common.authorize(auth,path,"train")

def test_exclusive_atomic_publish(tmp_path):
    path=tmp_path/"x.json"; common.publish(path,dict(x=1))
    with pytest.raises(FileExistsError): common.publish(path,dict(x=2))
    assert common.read(path)=={"x":1} and not list(tmp_path.glob(".*.tmp"))

def test_whole_toy_matrix_seals_before_test(toy,tmp_path):
    out=tmp_path/"execution"; out.mkdir(); runner.train_matrix(toy,out)
    assert common.read(out/"all_checkpoints_sealed.json")["all_models"]==12
    vault=common.open_vault(toy,out,common.read(toy)["cases"][0]); assert set(vault)=={"allowed","test_nodes","test_labels"}
    file=next((out/"training").glob("*/*/selected_state.pt"))
    with file.open("ab") as f: f.write(b"tamper")
    with pytest.raises(RuntimeError,match="frozen file mismatch"): common.open_vault(toy,out,common.read(toy)["cases"][0])

def test_forced_failure_retains_partial_not_seal(toy,tmp_path):
    out=tmp_path/"interrupted"; out.mkdir()
    class Interrupted:
        def begin(self,*a): pass
        def end(self,*a): pass
        def __call__(self,event):
            if isinstance(event,dict) and event.get("phase")=="round_end": raise RuntimeError("engineering forced failure")
    with pytest.raises(RuntimeError,match="engineering forced failure"): runner.train_matrix(toy,out,progress=Interrupted())
    assert list(out.glob("training/*/*/failure.json")) and list(out.glob("training/*/*/round_001.json"))
    assert not (out/"all_checkpoints_sealed.json").exists()

def test_hidden_test_label_changes_cannot_enter_client_payload():
    import torch
    x=torch.randn(12,2); y=torch.arange(12)%2; owners=torch.arange(12)%3
    edges=torch.tensor([[0,1],[1,0]]); masks=tuple(torch.arange(12)%4==i for i in range(4))
    a=common._client_payload(x,edges,owners,y,masks); changed=y.clone(); changed[masks[1]|masks[3]]=1-changed[masks[1]|masks[3]]
    b=common._client_payload(x,edges,owners,changed,masks)
    assert all(torch.equal(first[k],second[k]) for first,second in zip(a,b) for k in first)

@pytest.mark.parametrize("fault",["rss","timeout","record","storage"])
def test_supervisor_budget_failures_are_retained(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    import time
    import json
    out=tmp_path/"budget"
    class FakeProcess:
        pid=123456789; returncode=None
        def poll(self): return self.returncode
        def wait(self,timeout=None): self.returncode=-15; return self.returncode
    process=FakeProcess()
    def launch(*args,**kwargs):
        if fault=="record":
            (out/"events.jsonl").write_text(json.dumps(dict(phase="budget_begin",scope="record",seconds=0,monotonic=time.monotonic()-1))+"\n")
        return process
    monkeypatch.setattr(runner.subprocess,"Popen",launch)
    fake=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=(3*1024**3 if fault=="rss" else 10)),children=lambda recursive:[])
    monkeypatch.setattr(runner.psutil,"Process",lambda pid:fake)
    killed=[]; monkeypatch.setattr(runner.os,"killpg",lambda *args:killed.append(args))
    if fault=="timeout":
        now=time.monotonic(); calls=iter([now,now+901,now+902,now+903]); monkeypatch.setattr(runner.time,"monotonic",lambda:next(calls))
    if fault=="storage":
        large=SimpleNamespace(is_file=lambda:True,stat=lambda:SimpleNamespace(st_size=11*1024**3))
        monkeypatch.setattr(Path,"rglob",lambda *args:[large])
    args=SimpleNamespace(action="dry-run",out=str(out),manifest=None,authorization=None)
    with pytest.raises(SystemExit,match="STOP"): runner.supervise(args)
    assert killed and (out/"dry-run_supervisor_failure.json").is_file()
    assert common.read(out/"dry-run_resources.json")["stop_reason"] is not None
