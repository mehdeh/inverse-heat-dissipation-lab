## inverse-heat-dissipation-lab

Status: WIP (work in progress) — experimental/draft code and notebooks. Expect frequent breaking changes, renames, or removals until stabilization.

### Overview
This repository contains experimental extensions, refactors, and notebooks built on top of the inverse heat dissipation generative modeling framework. The goal is to iterate quickly on ideas (including exploratory notebooks) and converge towards a cleaned-up, reproducible codebase over time.

### Upstream Base and Attribution
Base code for this repository is derived from the following upstream project (MIT license). We explicitly acknowledge and reference the original work:

- AaltoML — Generative Modelling With Inverse Heat Dissipation: [github.com/AaltoML/generative-inverse-heat-dissipation](https://github.com/AaltoML/generative-inverse-heat-dissipation)

Please see the upstream repository for original implementation details, training/evaluation instructions, and references. We retain the upstream LICENSE terms (MIT) and acknowledge the original authors. If you use or distribute this repository, keep the MIT license notice intact and include clear attribution to the upstream project.

### What’s in this repo (draft)
- Added exploratory notebooks in `notebooks/` for rapid experimentation and ablations.
- Minor modifications to training/evaluation code and configs to support experiments.
- Temporary scripts for local runs and sanity checks.

Note: These changes are experimental and may not be finalized. File names, APIs, and results can change without notice during the WIP phase.

### Getting Started (experimental)
- Environment: start from the upstream environment (see their `requirements.txt`) and install any extra local dependencies you need for notebooks.
- Data: follow the upstream data preparation instructions where applicable. Some experiments rely on torchvision auto-downloads (e.g., MNIST, CIFAR-10).
- Running: for now, refer to the upstream README for installation/training/sampling/evaluation and adapt paths/configs as needed for the experimental additions here.

Given the draft status, we do not yet guarantee reproducibility or backward compatibility. Use at your own risk during this phase.

### Licensing
- Upstream base code: Copyright of the original authors, used under the MIT license. The MIT license text and notices from the upstream project are retained.
- Upstream project: [github.com/AaltoML/generative-inverse-heat-dissipation](https://github.com/AaltoML/generative-inverse-heat-dissipation)
- Additions in this repository (new notebooks, scripts, and modifications authored here) are dedicated to the public domain under CC0 1.0. You may copy, modify, and use them for any purpose without restriction. CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/

### Citation
If you build upon the original method, please cite the upstream paper as requested by the authors:

```bibtex
@inproceedings{rissanen2023generative,
  title={Generative modelling with inverse heat dissipation},
  author={Severi Rissanen and Markus Heinonen and Arno Solin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

### Acknowledgements
We thank the authors of the upstream repository for releasing their code under MIT, which made these experiments possible. For details about the original approach, datasets, and metrics, see the upstream README and documentation: [github.com/AaltoML/generative-inverse-heat-dissipation](https://github.com/AaltoML/generative-inverse-heat-dissipation).

