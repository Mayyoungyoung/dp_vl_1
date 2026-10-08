from scripts.research_v3_prepare_edit_support import addition_status


def test_unknown_positive_is_added_only_in_registered_all_mode_control():
    assert addition_status('over',{'gap'},'path',set())=='unreferenced_positive'
    assert addition_status('over',{'gap'},'path',set(),True)=='added_new_mode'
    assert addition_status('over',{'gap'},'path',{'path'},True)=='duplicate'
    assert addition_status('gap',{'gap'},'path',set(),True)=='added'
    assert addition_status('gap',{'gap'},'path',{'path'})=='duplicate'
