# Pilot4 visual QA

All four actual front RGBs in `ALL4_RGB.png` and all twelve target sheets were visually inspected. Each target sheet retains all nine requested slots with XY and XZ projections, raw paths, collector H24, failures and fixed geometry. This is a descriptive review after collection; no path was edited or removed.

- `401000`: all three targets show accepted lateral/over routes, with broad detours and long arcs retained. Target1 slot8 and target2 slot6 exceed 2 m. Target2 slot6 is accepted unknown, not a fabricated extra known mode.
- `401001`: side proposals often terminate before the obstacle. Target1 slot7 has recorded arm-environment collision; target2 slot2 is a complete path rejected by H24 clearance. The accepted set is over-only in these sampled references.
- `401002`: all targets include gap0/gap1/gap2/over known positives. Target0 slot0 and target2 slot6 have conspicuous high arcs; target2 slot4 is rejected by clearance and slot8 remains accepted unknown. Raw/H24 overlays are consistent with the numerical verification.
- `401003`: many saved partial paths stop near the higher posts. Target0 slot3 records body collision. Target2 slot6 is a valid gap2 route with a large detour; the other accepted paths are over routes. Partial failures are plotted, not discarded.
- Four RGBs visibly contain the registered 1/1/2/2 posts and three colored targets. No missing image, blank rendering or obvious layout corruption was observed. Height and other layout attributes vary together; the higher-post failure concentration does not identify a causal effect of height alone.

Maximum accepted recorded height is 1.607 m. Large loops remain a data-efficiency risk, not grounds for post hoc filtering. Projection overlap with a box does not itself prove a 3D collision; the unchanged 3D checks provide the numerical decision. These figures do not certify continuous full-body collision freedom or an independently reloaded simulator scene.

## Inspected files

- [ALL4_RGB.png](quality_analysis/figures/ALL4_RGB.png) — SHA256 `6f0075dd3e7d2b78b09384f614a8d295115e1ea0cc9d0381a463423f4c811ea2`
- [layout_variation_401000_target0.png](quality_analysis/figures/layout_variation_401000_target0.png) — SHA256 `d026ed6b170287f28c75e4fbf010491a4f781b01890076e8e6b4c2c3f49dac01`
- [layout_variation_401000_target1.png](quality_analysis/figures/layout_variation_401000_target1.png) — SHA256 `86a746758601e5a3b13ecff3e989cd5c0f2263ffac1ead27e6cc9a78d22308ae`
- [layout_variation_401000_target2.png](quality_analysis/figures/layout_variation_401000_target2.png) — SHA256 `84a220b8e0a429abd2d4a40672e39312dc1eb5c521f9b9c8b9f864474c6bc16e`
- [layout_variation_401001_target0.png](quality_analysis/figures/layout_variation_401001_target0.png) — SHA256 `c4fc9fed2723aa7113394546d508f0aecb4f2098cd48f6ba1556850d358a4e9e`
- [layout_variation_401001_target1.png](quality_analysis/figures/layout_variation_401001_target1.png) — SHA256 `e5fdf2e12147878161521c8abe8491b53bbb2365b2e8a6e1f0fb9dca3e21533f`
- [layout_variation_401001_target2.png](quality_analysis/figures/layout_variation_401001_target2.png) — SHA256 `b396457a2b1e4bdf3fbbd67be7007496bf356b8bbac24bcbf8568af906f2e593`
- [layout_variation_401002_target0.png](quality_analysis/figures/layout_variation_401002_target0.png) — SHA256 `ca09fbde20d40a1bcb9333c3deb7e1ac265fafb211b4ce4c0aef9c896a9b6c4c`
- [layout_variation_401002_target1.png](quality_analysis/figures/layout_variation_401002_target1.png) — SHA256 `ae7565c759fffc949f015a9690dd7419d837aaad817ca2ac6a891d2261733fab`
- [layout_variation_401002_target2.png](quality_analysis/figures/layout_variation_401002_target2.png) — SHA256 `ca51ab09a5d0561136de6c1b125d00c2792e2b2a2e704e54e041dfe47d503156`
- [layout_variation_401003_target0.png](quality_analysis/figures/layout_variation_401003_target0.png) — SHA256 `576a7e4a236a385ef26e2db7d36d3d002bb97d7265c5f68972abe938f8f878b2`
- [layout_variation_401003_target1.png](quality_analysis/figures/layout_variation_401003_target1.png) — SHA256 `b127faf9669a64e2a1c2e7830e111de68999c83002cfec78affc2f804aacb793`
- [layout_variation_401003_target2.png](quality_analysis/figures/layout_variation_401003_target2.png) — SHA256 `d526fbfaf44d53e15082757039fd672822a787cfeae85d0bdf8f7aadabcdb4fa`
