# Data provenance and preparation boundary

The frozen task objects use 2024 data at 15-minute resolution for Los Angeles,
San Diego, Fresno, and Sacramento. Solar inputs identify NREL NSRDB GOES CONUS
v4.0.0; load inputs identify the NREL OEDI ComStock AMY2018 release 1
large-office aggregate; prices identify CAISO OASIS day-ahead LMP at
`TH_NP15_GEN-APND`. Exact source identifiers and location parameters are in
`frozen_inputs/project/configs/frozen_protocol.yaml`.

The public reproduction consumes the exact retained vectors embedded in frozen
tasks and never downloads data. This makes the paper analyses reproducible
without network access. It does not recreate every candidate considered before
selection because complete PV, load, and price vectors for rejected candidates
were not retained. Admission results and selection traces are preserved, but
that historical retention boundary prevents a full alternative-selection-rule
counterfactual.

Source datasets retain their upstream licenses and terms. They are not
relicensed by this package.
