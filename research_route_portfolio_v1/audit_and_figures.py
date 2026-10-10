"""Local artifact audit and plots from actual sealed development predictions."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[1]
STUDY=WORKSPACE/'research_route_portfolio_v1'
RUN=WORKSPACE/'runs/route_portfolio_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    jobs=[read(p) for p in (WORKSPACE/'runs/selective_repair_v1/jobs').glob('portfolio*/receipt.json')]
    assert len(jobs)==30 and all(j['status'] in ('completed','failed') for j in jobs)
    assert [j['id'] for j in jobs if j['status']=='failed']==['portfolio_shift_B0_seed0_v1']
    pools={};sources={};unknown=0
    archives={}
    for folder in sorted(RUN.glob('shift_*'))+sorted(RUN.glob('preference_C_*')):
        seal=read(folder/'PREDICTION_SEAL.json')
        expected=seal.get('predictions_sha256',seal.get('pool_sha256'))
        assert sha(folder/'pool.npz')==expected
        assert (folder/'RESULTS.json').exists() and not read(folder/'RESULTS.json')['locked_access']
        pools[folder.name]=dict(pool_sha256=expected,results_sha256=sha(folder/'RESULTS.json'),rows_sha256=sha(folder/'rows.json'))
        for path,h in seal['source_sha256'].items():
            key=seal['source_commit']+':'+path
            if key not in sources:
                commit=seal['source_commit']
                archive=WORKSPACE/'runs'/('route_portfolio_runtime_a299ca5.tar' if commit.startswith('a299ca5') else f'route_portfolio_runtime_{commit}.tar')
                with tarfile.open(archive) as tf:actual=tf.extractfile(path).read()
                canonical=subprocess.check_output(['git','show',key],cwd=WORKSPACE)
                assert hashlib.sha256(actual).hexdigest()==h,key
                assert actual.replace(b'\r\n',b'\n')==canonical.replace(b'\r\n',b'\n'),key
                sources[key]=dict(actual_export_sha256=h,canonical_git_sha256=hashlib.sha256(canonical).hexdigest())
                archives[commit]=dict(file=str(archive.relative_to(WORKSPACE)),sha256=sha(archive))
        if folder.name.startswith('shift_'):
            rows=read(folder/'rows.json')
            unknown+=sum(v and w is None for r in rows for v,w in zip(r['valid'],r['words']))
            assert len(rows)==336 and len({r['family'] for r in rows})==16
            assert all(sum(r['valid'])-r['raw']['distinct']==r['duplicate_valid_routes'] for r in rows)
    assert len(pools)==26 and unknown==0
    replay=read(RUN/'REPLAY_V1/RESULTS.json')
    assert sha(RUN/'REPLAY_V1/sampling_rng.pt')==replay['rng_state_sha256']
    assert replay['stochastic_paths_events_q_modes_indices_exact'] and replay['preference_q_max_abs_error']==0.
    report=dict(jobs=len(jobs),successful_jobs=29,failed_jobs=1,elapsed_command_seconds=sum(j['elapsed_seconds'] for j in jobs),
        active_jobs=[],actual_commands={j['id']:dict(command=j['command'],source_commit=j['source_commit'],status=j['status'],elapsed_seconds=j['elapsed_seconds']) for j in jobs},
        preserved_failure=dict(job='portfolio_shift_B0_seed0_v1',reason='Checker assumed list indices had tolist; all336 predictions were already sealed. Recovered checking only; no generation rerun.'),
        export_failure=dict(reason='Interrupted oversized full-repository transfer; partial tar failed extraction before launch. Preserved failed export; replaced by scoped runtime archive in a fresh release directory.',
            failed_server_release='research_v2/releases/a299ca5868d03393748d42897cfff73e32612c62_failed_full_export',experiment_seconds=0),
        material_archive_sha256=sha(WORKSPACE/'runs/route_portfolio_material_20261010_v1.tar.gz'),
        replay_archive_sha256=sha(WORKSPACE/'runs/route_portfolio_replay_20261010_v1.tar.gz'),
        pools=pools,actual_source_hashes_verified_against_archives=sources,source_archives=archives,
        canonical_git_content_equal_after_eol_normalization=True,unclassified_valid_routes=unknown,
        replay=replay,cpu_tests=14,new_training_updates=0,new_collection=0,locked_access=False,
        prototype_core_supported=True,submission_ready=False,
        scope='Retrospectively chosen paper direction, with complete frozen development comparisons and disclosed tradeoffs. Historical conjunctive gates not relabeled.')
    (STUDY/'results/ARTIFACT_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    out=STUDY/'figures';out.mkdir(exist_ok=True)
    main=read(STUDY/'results/shift_statistics_v1/RESULTS.json')
    names=['B0','Bset','C_ordinary','C','C0_success'];labels=['Ordinary\nmatching','All-positive\nmatching','Random\nsampling','Mode\nportfolio','C0 + success\n(5 head seeds)']
    colors=['#a6aebb','#8b96a7','#e8b26c','#437eb4','#4c9d83']
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    for a,metric,title in zip(ax,['U8','V4'],['Distinct valid modes among 8 generated routes','Validity among 4 selected routes']):
        vals=[main['models'][n]['mean'][metric] for n in names]
        if metric=='V4':vals=[100*v for v in vals]
        bars=a.bar(np.arange(5),vals,color=colors,width=.7)
        a.set_xticks(np.arange(5),labels);a.set_title(title,pad=12);a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
        for b,v in zip(bars,vals):a.text(b.get_x()+b.get_width()/2,b.get_height()+(.1 if metric=='U8' else .6),f'{v:.2f}',ha='center',fontsize=10)
        a.set_ylim(0,8 if metric=='U8' else 100)
    fig.suptitle('Different rendered DEV families: 16 families, 336 requests',y=1.01)
    fig.tight_layout();fig.savefig(out/'coverage_and_return.png',dpi=180,bbox_inches='tight');fig.savefig(out/'coverage_and_return.svg',bbox_inches='tight');plt.close(fig)
    pref=read(STUDY/'results/preference_statistics_v1/RESULTS.json')['preferences'];names=['ground_only','over_any','first_gap0']
    labels=['No over\npassage','At least one\nover passage','First row\nuses gap 0']
    fig,ax=plt.subplots(1,2,figsize=(12,4.3));x=np.arange(3)
    for a,metric,title in zip(ax,['compliant_valid_fraction','any_compliant_valid'],['Fraction of returned routes valid and compliant','Requests with at least one valid compliant return']):
        for offset,arm,label,color in [(-.18,'baseline4','Select after generating 8','#8b96a7'),(.18,'proposed4','Allocate before generating 8','#437eb4')]:
            vals=[100*pref[n]['means'][arm][metric] for n in names]
            bars=a.bar(x+offset,vals,.34,color=color,label=label)
            for b,v in zip(bars,vals):a.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{v:.1f}',ha='center',fontsize=9)
        a.set_xticks(x,labels);a.set_title(title,pad=12);a.set_ylim(0,105);a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
    ax[0].legend(loc='upper center',bbox_to_anchor=(1.05,-.13),ncol=2,frameon=False)
    fig.tight_layout();fig.savefig(out/'preference_tradeoff.png',dpi=180,bbox_inches='tight');fig.savefig(out/'preference_tradeoff.svg',bbox_inches='tight');plt.close(fig)
    make_case(out)
    for path in out.glob('*.svg'):
        # Matplotlib emits trailing spaces in path data; normalize export text.
        path.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in path.read_bytes().splitlines())+b'\n')
    figure_manifest=dict(source_sha256=sha(Path(__file__)),
        files={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.svg')},
        actual_example_id='fresh_fs_family_641000_open_target0',selection_rule='First registered query and C0; no favorable-case selection',
        prediction_inputs={name:pools[name]['pool_sha256'] for name in ['shift_C_seed0','preference_C_seed0']},
        ground_truth_scope='Evaluation-only obstacle visualization; never a model input',locked_access=False)
    (out/'MANIFEST.json').write_text(json.dumps(figure_manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('jobs','successful_jobs','failed_jobs','elapsed_command_seconds','cpu_tests','prototype_core_supported','submission_ready')}))


def make_case(out):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scripts.evaluate_paired_modes import check_candidates
    from scripts.research_v3_audit import mode
    from scripts.analyze_paired_selection import select
    from research_route_portfolio_v1.preference_evaluate import preferences
    # VOCAB import would require local Torch; the vocabulary itself is fixed.
    import itertools
    vocab=['|'.join(w) for w in itertools.product(('gap0','gap1','gap2','over'),repeat=2)]
    def local(path):return WORKSPACE/Path(path).relative_to('/home/wzy/dpvlm/route_set_v1')
    # First registered query, fixed before rendering; no success-case selection.
    data=WORKSPACE/'data/feasible_space_generalization_v1/export'
    obs=json.loads((data/'observations.jsonl').read_text().splitlines()[0]);label=json.loads((data/'supervision.jsonl').read_text().splitlines()[0])
    cfg=read(local(label['route_config']))
    with np.load(local(label['verification_only'])) as z:truth={k:z[k] for k in ('obstacle_centers','obstacle_halfsizes')}
    with np.load(local(label['observation'])) as z:current={k:z[k] for k in ('gripper_pose','gripper_open')}
    with np.load(RUN/'shift_C_seed0/pool.npz') as z:base={k:z[k][0] for k in z.files if k!='ids'}
    with np.load(RUN/'preference_C_seed0/pool.npz') as z:
        j=list(z['preference_names']).index('over_any');p=z['paths'][0,j];e=z['events'][0,j];chosen=z['selected_indices'][0,j]
    wanted=preferences(vocab)['over_any'];restricted=base['q'].copy();restricted[[vocab[i] not in wanted for i in base['mode_ids']]]=-1.
    bp=select(base['paths'],restricted,4)
    fig=plt.figure(figsize=(14,4.5));a=fig.add_subplot(1,3,1);a.imshow(plt.imread(local(obs['image'])));a.set_title('Actual RGB input\nFirst registered scene, target 0');a.axis('off')
    for col,(paths,events,selected,title) in enumerate([(base['paths'],base['events'],bp,'Select after unconstrained generation'),(p,e,chosen,'Allocate for an over-passage preference')],2):
        a=fig.add_subplot(1,3,col,projection='3d');_,cc=check_candidates(paths,events,label,current,truth,cfg)
        for center,half in zip(truth['obstacle_centers'],truth['obstacle_halfsizes']):
            verts=np.array([center+half*np.array(v) for v in itertools.product((-1,1),repeat=3)])
            faces=[[verts[k] for k in f] for f in [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]]
            a.add_collection3d(Poly3DCollection(faces,facecolor='#949ca4',edgecolor='#737e89',alpha=.28,linewidth=.3))
        for i in selected:
            ok=cc[i]['TipValid'];word=mode(paths[i],cfg) if ok else None
            color='#30917e' if ok and word in wanted else ('#b74d44' if not ok else '#9b9b9b')
            a.plot(*paths[i].T,color=color,lw=2)
        target=np.array(label['semantic_targets']['centers'][0]);a.scatter(*target,color='#344d81',s=35)
        a.set_title(title);a.set_xlabel('x (m)');a.set_ylabel('y (m)');a.set_zlabel('z (m)');a.view_init(elev=22,azim=-62)
        a.set_xlim(.02,.62);a.set_ylim(-.3,.3);a.set_zlim(.75,1.25)
    fig.text(.52,.005,'Selected paths: green = valid and compliant, gray = valid outside preference, red = invalid. Obstacles are evaluation truth.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(out/'actual_preference_case.png',dpi=180,bbox_inches='tight');fig.savefig(out/'actual_preference_case.svg',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
