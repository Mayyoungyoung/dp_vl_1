import torch
from routeset.observed_probability import ProbabilisticGeometryRouteHead
from routeset import frozen_input_control as control


def test_frozen_encoders_keep_actual_anchor_and_context_while_routes_learn():
    torch.manual_seed(41)
    model=ProbabilisticGeometryRouteHead(feature_dim=16,width=16,point_width=8,max_candidates=8)
    inp=dict(features=torch.randn(2,16),current=torch.randn(2,8),world_xyz=torch.randn(2,20,3),
             rgb=torch.rand(2,20,3),uv=torch.rand(2,20,2),depth=torch.ones(2,20),valid_mask=torch.ones(2,20,dtype=torch.bool))
    model.eval()
    with torch.no_grad():before=model(**inp);direct=model.head.feature_encoder(inp['features'])+model.head.state_encoder(inp['current'])
    control.freeze(model);initial=control.digest(model)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003)
    for _ in range(3):
        model.train();control.freeze(model);p,e,g=model(**inp)
        opt.zero_grad(set_to_none=True);(p.square().mean()+e.mean()).backward();opt.step()
    model.eval()
    with torch.no_grad():after=model(**inp);newdirect=model.head.feature_encoder(inp['features'])+model.head.state_encoder(inp['current'])
    for key in ('anchor_xyz','context','attention'):torch.testing.assert_close(before[2][key],after[2][key],rtol=0,atol=0)
    torch.testing.assert_close(direct,newdirect,rtol=0,atol=0)
    assert not torch.equal(before[0],after[0])
    audit=control.audit(model,initial);assert 0<audit['trainable_parameters']<audit['total_parameters']
