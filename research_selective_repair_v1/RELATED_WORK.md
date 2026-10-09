# Direct precedents and falsification controls

Checked primary sources on 2026-10-09. The initially requested arXiv abstract
fetch had a network failure; HTML and publisher search then succeeded.

[Cascaded Diffusion Models for Neural Motion Planning](https://arxiv.org/html/2505.15157v1),
section IV-B, finds connected violating subtrajectories and applies its local
diffusion model once between nearby valid states. This directly precedes local
patching. Our proposed experiment concerns locating modifications from current
RGB-D while controlling damage to eight strong baseline routes; its distinction
requires measured same-information controls. We do not call a geometry-triggered
MLP a faithful implementation of that paper's diffusion architecture.

[DiffusionSeeder](https://proceedings.mlr.press/v270/huang25f.html) learns trajectory
initialization for motion optimization. Learned seeds or correction coordinates
must demonstrate gains over ordinary seeds with the same solver; changing the
sampler name does not establish a contribution.

[Learning Large Neighborhood Search Policy for Integer Programming](https://proceedings.neurips.cc/paper_files/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html)
learns which variables to release before solver reoptimization. Learned edit
support is therefore an established design principle, not by itself novelty.

[Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)
keeps baseline behavior where offline evidence is insufficient under its own MDP
assumptions. Our finite RGB-D route repair does not inherit its safety theorem.

The decisive contrasts are same-data center updates, whole residuals, global
scaling, observed geometric triggers and finite local optimization, followed by
frozen new-family and full-arm evidence. If ordinary methods account for gains,
the claimed mechanism must change. No prior-art novelty is assumed in advance.
