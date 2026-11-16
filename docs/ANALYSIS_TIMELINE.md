# Analysis timeline

This is the order in which the analyses of the paper were planned and run, with the dates recorded in the project's
step documents (2026). It is provided because the paper's secondary and follow-up analyses were pre-specified after
earlier results on the same 15 people were known (Section 2.11 and Limitations of the paper). The step codes in brackets
are the internal names that also appear in the folder and file names of this repository (see the README).

| Date | What was done |
|---|---|
| 31 Oct | Review of earlier, unpublished work by the authors on the same data; first study plan. |
| 1 Nov | Protocol for the literature review. |
| 7 Nov | Literature review completed. Study plan revised: electrode selection by nested forward selection inside each leave-one-subject-out fold with the one-standard-error rule; smallest effect of interest fixed at 5 percentage points (later used as the equivalence margin). Data audit: fresh download of the 300 GDF files with checksums, event check, bad-channel and trial registry (one channel added for subject 4, imagery). |
| 8 Nov | Preprocessing [A.1] and the sanity check against Ofner et al. (2017) [A.2]. In the first imagery run, the virtual onset used class-specific latencies, which leaks label information; it was replaced by the person's mean latency, and a rule was added that imagery timing never depends on class labels. Electrode study [A.2b]. |
| 9 Nov | Own-calibration ceiling and a check of the Lee et al. (2022) trial split [A.3]; the four training sources with TS+LDA and its controls [A.4a]; unsupervised adaptation [A.6]; ShallowConvNet in the four cells and its controls [A.4b]. |
| 10 Nov | PhysioNet re-analyses: left vs right fist [A.7-1] and right fist vs rest [A.7-2a]; a second candidate dataset (Chang et al. 2025) was dropped because its released files contain no EOG [A.7-3]. Decision on the shape of the paper. Calibration curves with all windows [A.8]. Robustness analyses: equivalence tests, decision time, hit and false-alarm rates, rule-defined 10-20 site sets [A.9]. Deep-model calibration curve [A.10]: a first run with re-implemented code did not reproduce the earlier deep-model result and was discarded; the final run [A.10b] used the earlier code with 5 seeds, and a repeat with identical code gave the hit and false-alarm rates [A.10c]. |
| 12 Nov | **Protocol 1** (library size, operating points, window choice, ensemble; `protocols/PROTOCOL_1_...md`) approved before any of these analyses ran; then run [A.11–A.13]. First generation of the tables and a p-value audit. Independent review of the plan and all results by a separate AI model [R.1]. **Protocol 2** (own-calibration references, specificity, multiplicity, deep-model eye and frontal controls; `protocols/PROTOCOL_2_...md`) approved before it ran; then run [N1–N3]. Amendment 2 of Protocol 2 (calibration curves with windows from 1.0 s [N5]; details of the deep-model controls [N4]) approved after N1–N3 and before N4–N5; then run. |
| 13 Nov | First draft; independent review of the draft by a separate AI model [R.2]; correction of the same-budget comparison of the hybrid and own calibration (`N5_same_budget.csv`; no other result changed); final figure and table run (92 checks against the step documents passed; no conclusion changed in the p-value audit); second draft. |
| 14 Nov | Manuscript files built; wording edits after a check against the journal's guide for authors and an expert reading (no result changed). |
| 15 Nov | This repository: the preprocessing and the sanity check were rebuilt from the raw data and checked against the files used for the paper (README, "Verification"). |

Notes
- Protocol 2 names its planned output folder `16_N1_N4`; the results were saved in `15_R1_followup`.
- The two protocols are copied verbatim from the project record; they use the internal step codes and refer to
  internal review documents that are not part of this repository.
