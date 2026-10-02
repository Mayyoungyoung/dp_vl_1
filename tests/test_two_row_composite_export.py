import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import export_two_row_composite_observations as composite

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    """Real mechanical file/budget/hash paths, mock existing raw reader only.

    The old reader's RGBD/restore/positive validation is covered by its unchanged
    integration tests. This fixture isolates composite ordering and raw scope.
    """
    spec = json.loads((ROOT/'configs/observed_two_row_composite108_v1.json').read_text())
    sources, registrations, manifests, gates = {}, {}, {}, {}
    for name,count,seed,selected in [('old',116,283200,76),('extension',288,400000,32)]:
        source=tmp_path/name;source.mkdir();sources[name]=source;plans=[]
        for i in range(count):
            role=('TRAIN' if i<64 else 'DEV_MODEL' if i<76 else 'TEST_LOCKED') if name=='old' else ('TRAIN' if i<256 else 'DEV_MODEL')
            plans.append(dict(index=i,parent_id='two_row_reach_%d'%(seed+i),role=role,config=dict(split=role),
                registered_geometry_exact_sha256=sha(name+str(i)+'exact'),
                registered_geometry_1mm_sha256=sha(name+str(i)+'1mm'),registration_eligible=True))
        registration=dict(parent_plan=plans)
        write(source/'registration.json',registration)
        manifest=dict(protocol=name+'_test_corpus',registration_sha256=composite.digest(source/'registration.json'),source_sha256={})
        write(source/'corpus_manifest.json',manifest)
        registrations[name]=registration;manifests[name]=manifest;gates[name]=dict(blocked_parent_ids=[],mechanical_only=True)
        for p in plans[:selected]:
            present=not(name=='old' and p['index']==20)
            c=dict(parent_id=p['parent_id'],index=p['index'],role=p['role'],requested_routes=27,
                initial_observation_saved=present,completed_slots=0,attempted_lower=0,attempted_upper=0,
                unattempted_lower=27,unattempted_upper=27,mechanical_files_sha256={},
                actual_geometry_1mm_sha256=p['registered_geometry_1mm_sha256'] if present else None)
            write(source/'closures'/('%03d.json'%p['index']),c)
        # A forbidden raw open would fail JSON decoding; no raw exists for future TRAIN either.
        poison=source/'parents'/('TEST_LOCKED' if name=='old' else 'DEV_MODEL')/'sealed.json'
        poison.parent.mkdir(parents=True);poison.write_text('FORBIDDEN RAW')
    monkeypatch.setattr(composite.formal,'verify_corpus',lambda source:(registrations['old'],manifests['old']))
    monkeypatch.setattr(composite.extension,'verify_corpus',lambda source:(registrations['extension'],manifests['extension']))
    monkeypatch.setattr(composite.formal,'live_layout_gate',lambda source:gates['old'])
    monkeypatch.setattr(composite.extension,'live_layout_gate',lambda source:gates['extension'])
    reads=[]

    def reader(source,plan,closure,hashes):
        name='old' if Path(source)==sources['old'] else 'extension';reads.append((name,plan['index']))
        assert plan['index'] < (76 if name=='old' else 32)
        x=[];y=[]
        if closure['initial_observation_saved']:
            for t in range(3):
                x.append(dict(id=plan['parent_id']+'_target%d'%t,parent_id=plan['parent_id'],split=plan['role'],
                              image=str(source/'image.png'),instruction='Touch the color sphere.'))
                y.append(dict(id=x[-1]['id'],parent_id=plan['parent_id'],split=plan['role'],routes=[],
                              reference_set_complete=False,route_types=[]))
        inventory=dict(parent_id=plan['parent_id'],index=plan['index'],split=plan['role'],requested_inputs=3,
            requested_routes=27,closure=closure,observed_inputs=len(x),positive_references=0,
            unobserved_requested_inputs=3-len(x),inputs_without_reference=len(x))
        return x,y,[],inventory

    monkeypatch.setattr(composite.old_exporter,'read_parent',reader)
    old=tmp_path/'old_export';old.mkdir();rows={name:[] for name in composite.JSONL_FILES};parents=[]
    for plan in registrations['old']['parent_plan'][:76]:
        c=json.loads((sources['old']/'closures'/('%03d.json'%plan['index'])).read_text())
        x,y,z,p=reader(sources['old'],plan,c,{})
        for name,data in zip(composite.JSONL_FILES,(x,y,z)):rows[name].extend(data)
        parents.append(p)
    for name,data in rows.items():
        # Deliberately use noncanonical whitespace + CRLF. Export must not normalize it.
        (old/name).write_bytes(''.join(json.dumps(r,separators=(', ', ' : '))+'\r\n' for r in data).encode())
    write(old/'parent_inventory.json',parents)
    old_selection=json.loads((ROOT/'configs/observed_two_row_prefix76_selection_v1.json').read_text())
    old_manifest=dict(protocol=composite.old_exporter.PROTOCOL,selection=old_selection,source_dataset=str(sources['old']),
        actual_inputs=225,requested_parents=76,source_files_sha256={},
        output_files_sha256={name:composite.digest(old/name) for name in composite.OUTPUT_FILES})
    write(old/'export_manifest.json',old_manifest)
    # This fixture models its own historical hash; production constant remains pinned.
    monkeypatch.setattr(composite,'OLD_MANIFEST_SHA256',composite.digest(old/'export_manifest.json'))
    spec['old_export_manifest_sha256']=composite.OLD_MANIFEST_SHA256
    def old_verify(path):
        assert Path(path)==old
        for name,value in old_manifest['output_files_sha256'].items():
            if composite.digest(old/name)!=value:raise ValueError('Historical export changed')
        return old_manifest,gates['old']
    monkeypatch.setattr(composite.old_exporter,'verify_export',old_verify)
    reads.clear()
    return dict(sources=sources,registrations=registrations,gates=gates,spec=spec,old=old,
                output=tmp_path/'export',reads=reads,rows=rows)


def build(c):
    return composite.build(c['sources']['old'],c['sources']['extension'],c['old'],c['spec'],c['output'])


def closure(c,name,index):
    path=c['sources'][name]/'closures'/('%03d.json'%index)
    return path,json.loads(path.read_text())


def test_fixed_export_preserves225_byte_prefix_and_missing_denominators(corpora):
    c=corpora;m=build(c)
    assert (m['requested_parents'],m['requested_inputs'],m['requested_routes'])==(108,324,2916)
    assert m['actual_inputs']==321 and m['unobserved_requested_inputs']==3
    assert m['actual_inputs_by_role']==dict(TRAIN=285,DEV_MODEL=36)
    assert m['actual_observed_train_parents']==95
    assert len(m['extension_added_input_ids'])==96
    assert c['reads']==[('old',i) for i in range(76)]+[('extension',i) for i in range(32)]
    for name in composite.JSONL_FILES:
        assert (c['output']/name).read_bytes().startswith((c['old']/name).read_bytes())
    assert not m['extension_dev_raw_opened'] and not m['cache_generated'] and not m['training_authorized']
    composite.verify_export(c['output'])


@pytest.mark.parametrize('name,index',[('old',75),('extension',31)])
def test_whole108_closure_gate_before_any_raw(corpora,name,index):
    c=corpora;closure(c,name,index)[0].unlink()
    with pytest.raises(ValueError,match='fully closed'):build(c)
    assert not c['reads'] and not c['output'].exists()


@pytest.mark.parametrize('kind',['exact','1mm'])
@pytest.mark.parametrize('other_index',[0,256])
def test_cross_corpus_registered_duplicate_closes_all_members_without_raw(corpora,kind,other_index):
    c=corpora;old=c['registrations']['old']['parent_plan'][0];new=c['registrations']['extension']['parent_plan'][other_index]
    new['registered_geometry_%s_sha256'%kind]=old['registered_geometry_%s_sha256'%kind]
    _,gate=composite.mechanical_gate(c['sources']['old'],c['sources']['extension'],c['spec'])
    assert dict(corpus='old',parent_id=old['parent_id']) in gate['blocked_members']
    assert dict(corpus='extension',parent_id=new['parent_id']) in gate['blocked_members']
    with pytest.raises(ValueError,match='layout gate blocked'):build(c)
    assert not c['reads']


def test_actual_geometry_mismatch_rejects_before_raw(corpora):
    c=corpora;p,r=closure(c,'extension',0);r['actual_geometry_1mm_sha256']='a'*64;write(p,r)
    with pytest.raises(ValueError,match='layout gate blocked'):build(c)
    assert not c['reads']


@pytest.mark.parametrize('mutation',['role','filename','hash','budget','missing_actual'])
def test_bad_closure_proof_refused_before_raw(corpora,mutation):
    c=corpora;p,r=closure(c,'extension',1)
    if mutation=='role':r['role']='DEV_MODEL'
    elif mutation=='hash':r['mechanical_files_sha256']={'mechanical_receipt.json':'a'*64}
    elif mutation=='budget':r['attempted_upper']=28
    elif mutation=='missing_actual':r['actual_geometry_1mm_sha256']=None
    write(p,r)
    if mutation=='filename':p.rename(p.with_name('1000.json'))
    with pytest.raises((ValueError,FileNotFoundError)):build(c)
    assert not c['reads']


def test_new_missing_input_and_partial_slot_uncertainty_retained(corpora):
    c=corpora;p,r=closure(c,'extension',1)
    r.update(initial_observation_saved=False,actual_geometry_1mm_sha256=None,
             attempted_lower=0,attempted_upper=1,unattempted_lower=26,unattempted_upper=27)
    write(p,r);m=build(c)
    assert m['actual_inputs']==318 and m['unobserved_requested_inputs']==6
    parents=json.loads((c['output']/'parent_inventory.json').read_text())
    assert len(parents)==108 and parents[77]['closure']['attempted_upper']==1
    assert parents[77]['requested_routes']==27


@pytest.mark.parametrize('key,value',[('extension_indices',list(range(1,33))),('old_indices',list(range(75))+[100]),
    ('extension_dev_raw_allowed',True),('failed_parent_replacements',1),('requested_routes',2889),
    ('cache_or_training_authorized',True)])
def test_prospective_scope_cannot_expand(corpora,key,value):
    c=corpora;c['spec'][key]=value
    with pytest.raises(ValueError,match='prospectively fixed'):build(c)
    assert not c['reads']


def test_revalidated_old_labels_must_equal_original(corpora,monkeypatch):
    c=corpora;reader=composite.old_exporter.read_parent
    def changed(source,plan,closed,hashes):
        x,y,z,p=reader(source,plan,closed,hashes)
        if y:y[0]['route_types']=['wrong']
        return x,y,z,p
    monkeypatch.setattr(composite.old_exporter,'read_parent',changed)
    with pytest.raises(ValueError,match='Revalidated old'):build(c)
    assert not c['output'].exists()


def test_historical_hash_or_bytes_change_refused(corpora):
    c=corpora;(c['old']/'supervision.jsonl').write_bytes(b'changed\n')
    with pytest.raises(ValueError,match='Historical export changed'):build(c)
    assert not c['reads']


def test_revalidation_cannot_overwrite_historical_source_hash(corpora,monkeypatch):
    c=corpora;original_verify=composite.old_exporter.verify_export
    path=str(c['sources']['old']/'registration.json')
    def verify(root):
        m,g=original_verify(root);m['source_files_sha256'][path]=composite.digest(path);return m,g
    original_reader=composite.old_exporter.read_parent
    def reader(source,plan,closed,hashes):
        result=original_reader(source,plan,closed,hashes);hashes[path]='a'*64;return result
    monkeypatch.setattr(composite.old_exporter,'verify_export',verify)
    monkeypatch.setattr(composite.old_exporter,'read_parent',reader)
    with pytest.raises(ValueError,match='Historical raw source hash'):build(c)
    assert not c['output'].exists()


def test_post_export_source_and_live_gate_rechecked(corpora):
    c=corpora;build(c)
    c['gates']['extension']['blocked_parent_ids']=[c['registrations']['extension']['parent_plan'][0]['parent_id']]
    with pytest.raises(ValueError,match='layout gate blocked'):composite.verify_export(c['output'])
    c['gates']['extension']['blocked_parent_ids']=[]
    p,r=closure(c,'extension',1);r['worker_elapsed_seconds']=123;write(p,r)
    with pytest.raises(ValueError,match='Closed source changed'):composite.verify_export(c['output'])


def test_existing_output_or_staging_never_overwritten(corpora):
    c=corpora;c['output'].with_name(c['output'].name+'.staging').mkdir()
    with pytest.raises(FileExistsError,match='Fresh'):build(c)
    assert not c['reads']


@pytest.mark.parametrize('field,value',[('requested_train_parents',95),('actual_observed_train_parents',96),
    ('training_authorized',True),('positive_references',1),('extension_added_input_ids',[]),
    ('reused_dev_input_ids',[])])
def test_verify_rejects_mutated_receipt_counts_and_scope(corpora,field,value):
    c=corpora;m=build(c);m[field]=value;write(c['output']/'export_manifest.json',m)
    with pytest.raises(ValueError):composite.verify_export(c['output'])


def test_symlinked_raw_role_root_refused_before_reader(corpora,monkeypatch):
    c=corpora;original=Path.is_symlink
    def flagged(path):
        return path==c['sources']['old']/'parents'/'TRAIN' or original(path)
    def forbidden_old_verify(*args):
        raise AssertionError('Historical verifier must not hash raw before the root guard')
    monkeypatch.setattr(Path,'is_symlink',flagged)
    monkeypatch.setattr(composite.old_exporter,'verify_export',forbidden_old_verify)
    with pytest.raises(ValueError,match='symlink'):build(c)
    assert not c['reads']


def test_sealed_role_raw_never_opened(corpora,monkeypatch):
    c=corpora;original=Path.open
    def guarded(path,*args,**kwargs):
        normalized=path.as_posix()
        if '/extension/parents/DEV_MODEL/' in normalized or '/old/parents/TEST_LOCKED/' in normalized:
            raise AssertionError('Forbidden raw read')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    build(c);composite.verify_export(c['output'])


def test_real_predeclared_registration_metadata_has_no_duplicate_selected_groups():
    old=json.loads((ROOT/'configs/observed_two_row_formal116_registered_v1.json').read_text())
    new=json.loads((ROOT/'configs/observed_two_row_extension288_registered_v1.json').read_text())
    for kind in ('exact','1mm'):
        by_hash={}
        for name,value in [('old',old),('extension',new)]:
            for p in value['parent_plan']:
                by_hash.setdefault(p['registered_geometry_%s_sha256'%kind],[]).append((name,p['index']))
        selected={('old',i) for i in range(76)}|{('extension',i) for i in range(32)}
        assert not [members for members in by_hash.values() if len(members)>1 and selected.intersection(members)]
