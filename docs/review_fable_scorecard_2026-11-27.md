# Supervisor review of the scorecard spec v1 (2026-11-27)

This is a read-only review of `docs/SCORECARD_spec.md` v1, done by a Fable supervisor. Every item is addressed in v2 (see the spec's change log).

## Critical

- **C1. Time readable by sequence models.**
  - The twin rule covered only wind and terrain. Level-2 sequence models can read hour and date from position within a day, because scheduled bursts sit on a fixed clock.
  - Fix: a time/position rule. G2, G3, G4, G5, G7 (other season) and G8 are scored on embeddings without that information.
- **C2. G7 injection set depended on the model.**
  - "Mix" was defined from K=6 states, so the test set differed per design or was separable by construction.
  - Fix: model-free mixes (the GPS-only labels) and frozen host and donor lists.
- **C3. G7 mean of three recalls was dominated by shuffled order.**
  - Shuffled-order recall rises mechanically with neighbour or sequence training.
  - Fix: separate rows; shuffle is descriptive; unseen-mix minus same-mix is the identity-corrected score.
- **C4. 2 paired sd over 3 seeds is a weak threshold.**
  - It is about t > 3.46 on 2 degrees of freedom, so about 3.7% false positives per row and direction. That gives about 23% chance wins over the selection rows and about 36% chance vetoes. Bird sampling noise is ignored.
  - Fix: 5 seeds, fixed per-row minimum effects, a paired bootstrap over birds, and absolute guard-rail tolerances.
- **C5. "model − handcrafted" was undefined across frames.**
  - The handcrafted identity score changes with the frame (0.259 against 0.178).
  - Fix: rank on the raw score, with a constant sensor-frame reference.

## Important

- **I1.** The G1 handcrafted reference was weak (AUC 0.42). Fix: add bird medians of continuous-mode flight speed, turn radius and turn rate, and use an equal-weight per-mode mean.
- **I2.** G3 is confounded with days since deployment, with which birds are present in each season, and possibly with accelerometer temperature offset. Fix: add these to the nuisance floor, and weight birds equally.
- **I3.** 20 locked birds cannot confirm bird-level rows (AUC standard error about 0.12). Fix: the locked set is confirmatory for window rows only. Small strata are handled by the method.
- **I4.** The birds used per row were unstated. Fix: stated.
- **I5.** G6 was ambiguous: speed channel, heading, class rule, threshold. Perched and take-off are rare. Fix: definitions, AUC as the metric, and a class count before freezing.
- **I6.** G4 was underspecified and measured seasonality. Fix: specified, and a G4b year-to-year row added.
- **I7.** G8 is mostly migrants against young dispersers. Fix: add young-tagged and winter window count to the floor, and evaluate the floor before scoring.
- **I8.** The G9 metric name did not match the code. Fix: balanced accuracy, with first-half/second-half time split.
- **I9.** The flight mask comes from r6 and carries recording-mode structure. Fix: state this.

## Minor

- G8b criterion.
- "Groups" may mean social groups to the owner.
- G5 confounding.
- Circular R² definition.
- A behaviour-type discovery row (G8c).
- Suspend a row when model minus floor is below the minimum effect.
- Fixed sampling seeds.
- r7_noneigh optimism on the search birds.
