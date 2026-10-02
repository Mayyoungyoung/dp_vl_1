from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from routeset.multigate import generate_dataset
from scripts.audit_multigate_closure_opportunity import analyze_parent, load_train_only
from scripts.audit_multigate_k2_closure import evaluate_k2, summarize, validate_training_config
from scripts.launch_multigate_k2_saturation import validate_registration


def registered():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/"configs/multigate_k2_saturation_budget_check_v1.json").read_text())


def test_registration_reuses_actual_historical_settings_and_own_k2_budget():
    config=registered()
    historical=dict(config["training"],candidates=4,dataset_sha256=config["data_sha256"])
    assert validate_registration(config,historical)["candidates"]==2
    for key,value in (("steps",4000),("width",128),("lr",.001),("candidates",2)):
        bad=deepcopy(historical);bad[key]=value
        with pytest.raises(ValueError):validate_registration(config,bad)
    bad=deepcopy(config);bad["gradient_target_slots"]*=2
    with pytest.raises(ValueError):validate_registration(bad,historical)


def test_checkpoint_must_be_genuinely_k2_trained_not_truncated_k4():
    config=registered()
    trained=dict(config["training"],horizon=24,cond_dim=34,dataset_sha256=config["data_sha256"],selection_split="DEV_MODEL")
    validate_training_config(trained,config)
    for key,value in (("candidates",4),("objective","subset"),("eval_every",250),("horizon",12)):
        bad=deepcopy(trained);bad[key]=value
        with pytest.raises(ValueError):validate_training_config(bad,config)


def test_new_k2_predictions_reuse_all_old_closures_and_no_solution_denominators(tmp_path):
    file=generate_dataset(tmp_path/"fixture.npz",train=8,dev_model=1,dev_score=0,calibration=0,test_locked=0,ood_locked=0,seed=7)
    data=load_train_only(file,8)
    prior_parents,prior_closures,predictions=[],[],[]
    for i,value in enumerate(data["scenes"]):
        refs=data["paths"][i,data["path_mask"][i]]
        old=refs[np.arange(4)%len(refs)]
        p,c=analyze_parent(value,refs,data["modes"][i,data["path_mask"][i]],old,str(data["parent_ids"][i]),str(data["scene_ids"][i]),.25)
        prior_parents.append(p);prior_closures.extend(c)
        predictions.append(refs[list(p["selections"]["farthest"])])
    predictions=np.asarray(predictions)
    parents,closures=evaluate_k2(data,predictions,prior_parents,prior_closures)
    result=summarize(parents)
    assert len(closures)==len(prior_closures)
    assert result["actual_k2"]["any_valid_solvable_closures"]==1.
    assert result["paired_any_valid_k2_minus_control"]["reference_farthest_k2"]["mean"]==0.
    assert result["solvable_closures"]==sum(c["has_solution"] for c in prior_closures)
    with pytest.raises(ValueError,match="exactly two"):
        evaluate_k2(data,np.repeat(predictions,2,axis=1),prior_parents,prior_closures)
    with pytest.raises(ValueError,match="order changed"):
        evaluate_k2(data,predictions,list(reversed(prior_parents)),prior_closures)
    with pytest.raises(ValueError,match="duplicate"):
        evaluate_k2(data,predictions,prior_parents,prior_closures+prior_closures[:1])


def test_empty_r_less_than_k_conditional_summary_keeps_null():
    summary=summarize([])
    assert summary["evaluable_parents"]==0
    assert summary["actual_k2"]["any_valid_solvable_closures"] is None
