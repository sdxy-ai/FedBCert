import json
import numpy as np
import pytest
import torch
from scripts import fedbcert_rg1_training as rg

def _spec(name="P"): return dict(name=name,input_dim=2,num_classes=2,hidden_dim=3,rounds=2)
def _clients():
    return [dict(features=torch.tensor([[1.,0.],[0.,1.],[1.,.2],[.2,1.]]),local_edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]),train_targets=torch.tensor([0,1,-1,-1]),validation_targets=torch.tensor([-1,-1,0,1])) for _ in range(3)]
def _initial(spec):
    torch.manual_seed(91); return {k:v.clone() for k,v in rg.make_model(spec).state_dict().items()}

def test_two_round_toy_outputs_pairing_and_earliest_tie(tmp_path,monkeypatch):
    monkeypatch.setattr(rg,"_validation_score",lambda model,clients:.5); records={}; schedules={}
    for name in ("P","I","MLP"):
        events=[]; out=tmp_path/name
        records[name]=rg.train_one(_spec(name),_clients(),_initial(_spec(name)),123,out,checkpoint=events.append)
        record=records[name]; assert record["selected_round"]==1 and record["rounds_completed"]==2
        rows=json.loads((out/"validation_trajectory.json").read_text()); assert len(rows)==2 and rows[0]["is_new_best"] and not rows[1]["is_new_best"]
        assert record["selected_tensor_sha256"]==rows[0]["tensor_sha256"] and record["last_tensor_sha256"]==rows[1]["tensor_sha256"]
        assert len(list(out.glob("round_*.json")))==2 and not (out/"failure.json").exists()
        schedules[name]=[(e["round"],e["client"],e["rng_seed"]) for e in events if e["phase"]=="client_start"]
        with pytest.raises(FileExistsError): rg.train_one(_spec(name),_clients(),_initial(_spec(name)),123,out)
    assert records["P"]["initial_tensor_sha256"]==records["I"]["initial_tensor_sha256"]
    expected=[(r,c,int(np.random.SeedSequence([123,r,c]).generate_state(1)[0])) for r in (1,2) for c in range(3)]
    assert schedules["P"]==schedules["I"]==expected

@pytest.mark.parametrize("name",["I","MLP"])
def test_graph_independent_controls_force_identity(name):
    prepared=rg._prepare_clients(rg._spec(_spec(name)),_clients(),torch.device("cpu"))
    assert all(torch.equal(c["shift"].to_dense(),torch.eye(4)) for c in prepared)

def test_interface_rejects_test_labels_and_protocol_changes(tmp_path):
    clients=_clients(); clients[0]["test_targets"]=torch.tensor([-1,-1,0,1])
    with pytest.raises(ValueError,match="only local"): rg.train_one(_spec(),clients,_initial(_spec()),123,tmp_path/"forbidden")
    assert not (tmp_path/"forbidden").exists()
    with pytest.raises(ValueError,match="fixes lr"): rg.make_model(dict(_spec(),lr=.02))

def test_interruption_preserves_completed_round_and_partial_state(tmp_path,monkeypatch):
    monkeypatch.setattr(rg,"_validation_score",lambda model,clients:.5)
    def stop(event):
        if event["phase"]=="round_end": raise RuntimeError("toy-stop")
    out=tmp_path/"interrupted"
    with pytest.raises(RuntimeError,match="toy-stop"): rg.train_one(_spec(),_clients(),_initial(_spec()),123,out,checkpoint=stop)
    failure=json.loads((out/"failure.json").read_text()); row=json.loads((out/"round_001.json").read_text())
    partial=torch.load(out/"last_state_partial.pt",map_location="cpu",weights_only=True)
    assert failure["rounds_completed"]==failure["last_aggregated_round"]==1
    assert rg._tensor_hash(partial)==row["tensor_sha256"]==failure["partial_tensor_sha256"]
    assert not (out/"training_record.json").exists() and not (out/"round_002.json").exists()
