# context/

Source papers, kept locally for reference. The PDFs are **gitignored** — they are ~24 MB
and re-fetchable from arXiv.

| file | paper |
|---|---|
| `2506.09849v1.pdf` | **IntPhys 2** — arXiv:2506.09849. Four principles (Permanence, Immutability, Spatio-Temporal Continuity, Solidity); the debug/artifact split we copy in Part 2. |
| `2510.11512v3.pdf` | **LikePhys** — arXiv:2510.11512. 12 scenarios across four domains (Rigid Body, Continuum, Fluid, Optical). |

Re-fetch:

```bash
curl -L -o context/2506.09849v1.pdf https://arxiv.org/pdf/2506.09849v1
curl -L -o context/2510.11512v3.pdf https://arxiv.org/pdf/2510.11512v3
```

Each family records which prior-art cell it covers, as `intphys2` and `likephys` fields in
`physloc/taxonomy.py`. `python -m physloc.cli taxonomy` prints the taxonomy.
