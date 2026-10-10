"""Bounded current-observation API check; not matched-head or native evaluation."""
import argparse
from research_selective_repair_v1.io import ROOT,RUN,read,write

def run(name):
    from research_selective_repair_v1.body_deployment import bind,feedback
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);records=[]
    for label,forecast,folder in [('state','native','body_native_state_seed0_pilot_v1'),('confusion','confusion','body_native_binary_seed0_pilot_v1')]:
        decoder=name+'_'+label+'_decoder';data=name+'_'+label+'_feedback'
        bind(decoder,RUN/folder/'last.pt',RUN/'constraints_seed0_v1/last.pt',RUN/'calibrated_prototype_v1/PROTOTYPES.json',
            forecast=forecast,word_safe=True,recipe_support='prefix',
            confusion=RUN/'body_native_mode_recipe_confusion_v1/MODEL.json' if forecast=='confusion' else None)
        feedback(data,RUN/decoder/'last.pt',ROOT/'data/selective_repair_interventions_v1',limit=1)
        counts=read(RUN/data/'BODY_COUNTS.json');summary=read(RUN/data/'SUMMARY.json')
        assert counts['current_observation_only'] and summary['locked_access'] is False
        assert all(c['final_candidates']==8 and c['internal_options']==24 for c in counts['decode_counts'])
        records.append(dict(label=label,decoder=decoder,feedback=data,decoded_sets=summary['counts']['decoded_sets'],counts=counts))
    write(out/'SUMMARY.json',dict(records=records,scope='One TRAIN observation, current RGB-D/cached encoder API; no formal matched head, uncached latency, or actual native execution result',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
