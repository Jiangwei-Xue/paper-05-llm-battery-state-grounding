# Third-party materials and upstream terms

This inventory was derived from the distribution metadata and license files
actually bundled in this repository, together with the frozen data-provenance
configuration. A file-specific notice or upstream term controls wherever it
applies.

| Repository path | Package or source | Version or identifier | Upstream URL | License or governing terms | Included notice |
|---|---|---|---|---|---|
| `runtime/vendor/numpy-2.5.1.dist-info/` and `runtime/vendor/numpy/` | NumPy | 2.5.1 | https://numpy.org/ | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`, as declared by the bundled `License-Expression` | `METADATA`; `licenses/LICENSE.txt`; all 16 additional `License-File` entries declared by `METADATA`; package-level component notices |
| `runtime/vendor/pandas-3.0.3.dist-info/` and `runtime/vendor/pandas/` | pandas | 3.0.3 | https://pandas.pydata.org/ | BSD 3-Clause License, with incorporated third-party component notices | `METADATA`; `LICENSE` |
| `runtime/vendor/scipy-1.18.0.dist-info/` and `runtime/vendor/scipy/` | SciPy | 1.18.0 | https://scipy.org/ | BSD License, with incorporated bundled-component notices | `METADATA`; `LICENSE.txt`; component license files retained inside the package tree |
| `runtime/vendor/pyyaml-6.0.3.dist-info/` and `runtime/vendor/yaml/` | PyYAML | 6.0.3 | https://pyyaml.org/ | MIT License | `METADATA`; `licenses/LICENSE` |
| `runtime/vendor/python_dateutil-2.9.0.post0.dist-info/` and `runtime/vendor/dateutil/` | python-dateutil | 2.9.0.post0 | https://github.com/dateutil/dateutil | Dual License: Apache License 2.0 and BSD License, as stated in the bundled metadata and license file | `METADATA`; `LICENSE` |
| Retained solar vectors and source identifiers under `frozen_inputs/project/` | NREL NSRDB GOES CONUS | v4.0.0 | https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv | NREL API and NSRDB dataset terms | Source endpoint and identifiers in `frozen_inputs/project/configs/frozen_protocol.yaml`; provenance in `docs/data_provenance.md` |
| Retained load vectors and source identifiers under `frozen_inputs/project/` | NREL OEDI End-Use Load Profiles / ComStock | AMY2018 release 1, California large-office aggregate | https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_aggregates/by_state/state=CA/ca-largeoffice.csv | Original NREL OEDI / ComStock dataset terms and attribution requirements | Source URL and identifier in `frozen_inputs/project/configs/frozen_protocol.yaml`; provenance in `docs/data_provenance.md` |
| Retained price vectors and source identifiers under `frozen_inputs/project/` | CAISO OASIS day-ahead LMP | 2024, node `TH_NP15_GEN-APND` | https://oasis.caiso.com/oasisapi/SingleZip | CAISO OASIS terms of use and attribution requirements | Query endpoint and identifiers in `frozen_inputs/project/configs/frozen_protocol.yaml`; provenance in `docs/data_provenance.md` |
| `frozen_inputs/project/runs/experiments_v2/e1/e1_20260801T044500Z/records/` and `frozen_inputs/project/runs/experiments_v2/e2/e2_20260801T054708Z/records/` | Historical hosted-model requests, responses, and provider metadata | DeepSeek and Qwen3.6 Flash experiment records | Applicable model-provider documentation and terms | Preserved request/response records and provider metadata; no provider credential is included |
| `frozen_inputs/project/segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records/` | Hosted-model requests, responses, and provider metadata | F1 experiment records | Applicable model-provider documentation and terms | Preserved request/response records and provider metadata; no provider credential is included |
| `frozen_inputs/project/segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records/` | Hosted-model requests, responses, and provider metadata | F0 experiment records | Applicable model-provider documentation and terms | Preserved request/response records and provider metadata; no provider credential is included |
| `frozen_inputs/project/segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records/` | Hosted-model requests, responses, and provider metadata | Qwen3.7-Plus sensitivity-tier records | Applicable model-provider documentation and terms | Preserved request/response records and provider metadata; no provider credential is included |
| Bibliographic records, quotations, trademarks, and any copied or incorporated third-party content | Respective upstream sources | As identified in the corresponding file or citation | File-specific | File-specific or upstream terms | File-level credit, citation, or upstream notice where supplied |

For third-party materials, public data, provider records, or any material whose
authorization scope cannot be confirmed, no new license is granted by this
repository. Reuse requires consultation of the upstream license, dataset terms,
or provider terms.

## Bundled-distribution audit

- NumPy's 17 `License-File` entries declared in `METADATA` are present under
  `runtime/vendor/numpy-2.5.1.dist-info/licenses/`.
- pandas retains its bundled `METADATA` and consolidated `LICENSE`, including
  the notices for incorporated components.
- SciPy retains its bundled `METADATA`, consolidated `LICENSE.txt`, and
  component license files in the package tree.
- PyYAML retains the `License-File: LICENSE` identified by `METADATA`.
- python-dateutil retains the `License-File: LICENSE` identified by `METADATA`;
  that file describes its dual Apache-2.0 and BSD licensing.

The upstream files named above, not this summary, are authoritative for the
bundled distributions.
