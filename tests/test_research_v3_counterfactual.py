from scripts.research_v3_counterfactual import survival


def test_colliding_path_does_not_certify_mode_absence():
    result=survival([True,True,False],[False,True,True],['left','right','over'],[True],['left'])
    assert result['invalidated_source_routes']==1
    assert result['surviving_modes']==['right']
    assert result['lost_witness_modes']==['right']
    assert result['surviving_witness_modes']==1


def test_duplicates_are_single_witness_and_repaired_mode_counts_as_retained():
    result=survival([True,True],[True,True],['left','left'],[False,True],['right','left'])
    assert result['surviving_source_routes']==2
    assert result['retained_witness_modes']==1 and not result['lost_witness_modes']


def test_budget_bound_counts_invalid_and_classified_duplicates_but_not_unknown_valid():
    args=([True,True],[True,True],['left','over'])
    full=survival(*args,[True,True],['right',None])
    assert full['destination_invalid_or_duplicate_slots']==0 and full['slot_feasible_lost_modes']==0
    duplicate=survival(*args,[True,True,False],['right','right',None])
    assert duplicate['destination_invalid_or_duplicate_slots']==2 and duplicate['slot_feasible_lost_modes']==2
