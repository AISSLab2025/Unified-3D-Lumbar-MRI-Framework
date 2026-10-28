# Third-party components

`SwarmSpine_Inference/` incorporates the SpineSwarm chatbot runtime, adapted
for this inference API. Its original MIT notice is retained in
`SwarmSpine_Inference/LICENSE` (Copyright 2026 SpineSwarm Authors).
The license of this component does not select a license for the entire repository.
Ollama and LLM weights are installed separately and are not redistributed here;
their respective licenses apply to local use.

`tools/roi/DINO/` contains DINO and Deformable DETR derived code. Retain the
copyright and license headers in those files. Upstream projects:

- DINO: https://github.com/IDEA-Research/DINO (Apache-2.0)
- Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR (Apache-2.0)
- Ultralytics: https://github.com/ultralytics/ultralytics (review its applicable AGPL/enterprise terms)
- Mamba: https://github.com/state-spaces/mamba

Before publication, include the applicable upstream license/NOTICE texts for
redistributed source and confirm the redistribution terms for trained weights.
No license for the authors' original code has been selected by this preparation.
The repository owner must select the intended license and add a root LICENSE.
Dependency licenses remain applicable independently of the original-code license.
