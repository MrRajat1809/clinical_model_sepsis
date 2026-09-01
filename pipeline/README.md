## Title

```
ClinicalTensorSepsis - Harmonised Sepsis-3 Cohorts with Hourly Physiology Derived from MIMIC-IV-Ext and eICU-CRD
```

---

## Version

```
1.0.0
```

---

![Figure 1: Analysis Pipeline](../Figures/Fig1.png)

## Abstract

Sepsis is the leading cause of in-hospital mortality worldwide, and prognostic
models built on one intensive care database frequently degrade when applied to
another. Progress on that problem is limited by the absence of cohorts that are
constructed identically across sources, because differences in cohort definition
are otherwise indistinguishable from differences in physiology.

ClinicalTensorSepsis provides two Sepsis-3 cohorts assembled under a single
specification from MIMIC-IV and the eICU Collaborative Research Database.
20,569 intensive care stays are included, each with thirty clinical variables placed 
on a common hourly grid spanning the first twenty-four hours after sepsis onset. 
Sepsis onset is defined as the coincidence of suspected infection with an acute 
increase of at least two points in the Sequential Organ Failure Assessment score, 
adjudicated by the same code in both databases.

The release contains observed measurements with their missingness preserved, a
reconstructed version produced by a transformer-based imputation model trained
only on the development cohort, a summary feature representation, and a version
of that representation aligned across databases by entropic optimal transport.
Recording density differs substantially between the two sources, and that
difference is documented per variable rather than hidden by imputation.

The resource supports work on prognostic modelling, missing-data methods,
domain adaptation, and cross-database transportability, and allows a model
developed on one database to be evaluated on the other without the confound of
differing cohort definitions.

---

## Background

Prognostic models for sepsis are usually developed and validated within a single
intensive care database. When such a model is applied to a different health
system its performance commonly falls, but the reason is rarely separable into
its parts. A drop may reflect genuine differences in case mix or physiology, or
it may reflect nothing more than the two cohorts having been defined by different
code, with different inclusion windows, different handling of missing organ
system data, and different comorbidity definitions.

Both MIMIC-IV [1] and the eICU Collaborative Research Database [2] are widely
used, and derived Sepsis-3 [3] cohorts exist for each. They are not, however,
constructed identically, so the comparison that matters most for transportability
work is the one that is hardest to make.

This resource was created to remove that confound. A single specification is
applied to both databases: the same inclusion criteria, the same suspected
infection window, the same SOFA component definitions and thresholds, the same
comorbidity algorithm, the same hourly aggregation rules, and the same feature
order. Where the two databases genuinely cannot support the same operational
definition, the difference is documented rather than smoothed over. The clearest
instance is the suspected infection time, which can be derived from the coupling
of microbiological culture with antibiotic administration in MIMIC-IV but must be
taken from recorded diagnoses in eICU-CRD.

The intention is that a difference in model performance between the two cohorts
can be attributed to the data rather than to the cohort builder.

---

## Content description

All files are gzip-compressed comma-separated values, formatted per RFC 4180,
except the data dictionary and cohort flow, which are uncompressed.

| File | Rows | Description |
| --- | --- | --- |
| `readme.txt` | — | package description and file summary |
| `cohort.csv.gz` | 20,569 | one row per intensive care stay |
| `hourly_observed.csv.gz` | 20,569 x 24 | measured values; blank means no observation |
| `hourly_imputed.csv.gz` | 20,569 x 24 | reconstructed values; complete |
| `features_model.csv.gz` | 20,569 | summary representation, 122 columns |
| `features_transported.csv.gz` | 20,569 | same representation after optimal transport |
| `data_dictionary.csv` | ~160 | per-variable documentation |
| `cohort_flow.csv` | ~15 | cohort attrition at each step |
| `sha256sums.txt` | — | checksums for every file |

Cohort sizes are 12,919 stays from MIMIC-IV and 7,650 from eICU-CRD.

Times are expressed as integer offsets in minutes from intensive care admission.
No absolute dates or timestamps are included in any file.

Source identifiers are retained so that users may join back to the original
databases, for which they will already hold credentialed access.

`hourly_observed.csv.gz` and `hourly_imputed.csv.gz` share a layout: one row per
stay-hour, with columns for source database, stay identifier, hour index from
zero to twenty-three, and the thirty clinical variables. The two files differ
only in whether unobserved cells are blank or filled.

`features_model.csv.gz` carries the one hundred and twenty-two summary features
together with the development and held-out partition assignment used in the
associated analysis, so that the split may be reproduced exactly.

`features_transported.csv.gz` carries the same columns after alignment, together
with a flag marking which rows were transported. Rows originating from MIMIC-IV
are unchanged; rows originating from eICU-CRD are mapped.

---

## Usage notes

### Execution workflow

To rebuild the released dataset from the source data, run:

```bash
bash pipeline/01_build_dataset.sh
```

The build runs the required dataset construction, harmonisation, and export
steps in dependency order. The released dataset is written to `release/`, and a
complete execution log is saved to `outputs/logs/build_dataset.log`.

The build script regenerates the released dataset; it does not reproduce the
full manuscript analysis.

### What this resource is useful for

The two cohorts are defined by one specification, so a difference in model
performance between them can be attributed to the data rather than to differing
cohort definitions. That makes the resource suited to work on external validation
and transportability, on domain adaptation between health systems, on missing
data methods for irregularly sampled clinical time series, and on prognostic
modelling in sepsis generally.

Releasing observed values alongside their missingness, rather than an imputed
matrix alone, is deliberate. Recording density differs markedly between the two
databases and that difference is itself a research object.

### Known limitations

Users should be aware of the following before drawing conclusions.

**Suspected infection is defined differently in the two databases.** MIMIC-IV
couples microbiology to antibiotic administration; eICU-CRD relies on recorded
diagnoses, because its microbiology and medication tables are too sparse to
support the coupling. This is the largest structural difference between the
cohorts and it cannot be removed without discarding one of them.

**Recording density differs substantially.** Per-variable density in each cohort
is given in the data dictionary. Urine output in particular is recorded far more
frequently in MIMIC-IV. Any analysis comparing the cohorts should account for
this, and analyses using the reconstructed file should note that a larger share
of the eICU values are reconstructed rather than measured.

**A SOFA component with no qualifying observation scores zero.** This is the
conventional operationalisation, but combined with the density difference it means
a variable recorded less often in one cohort will systematically contribute less
there.

**eICU-CRD has no absolute admission times.** First-stay ordering is proxied by
ascending stay identifier. This is a property of the source database.

**The transported file contains mapped values, not measurements.** Do not treat
eICU rows in `features_transported.csv.gz` as observed physiology.

**The Charlson Comorbidity Index relies heavily on text-based fallback in eICU-CRD.** In eICU-CRD, 
11.3% of diagnoses lack usable ICD codes and were scored via text matching. Furthermore, 51.9% of 
stays scored zero, suggesting potential undercoding rather than true absence of comorbidity.

### Complementary resources

The complete code that constructs this resource, including the exact queries and
the export script, is available [8]. The source databases are [1] and [2].

---

## Data access

This resource contains data derived from credentialed PhysioNet datasets. Access
to and use of the released dataset are subject to the PhysioNet Credentialed
Health Data License 1.5.0 and the PhysioNet Credentialed Health Data Use
Agreement 1.5.0.

Users must meet the applicable PhysioNet credentialing and data-use requirements
before accessing or using the credentialed data.

---

## Release Notes

**Version 1.0.0 (Initial Release)**

This is the initial release of the ClinicalTensorSepsis dataset, designed to support cross-database machine learning and transportability research. 

Key inclusions in this release:
* **Harmonized Sepsis-3 Cohorts**: 20,569 intensive care stays sourced from MIMIC-IV-Ext (12,919 stays) and eICU-CRD (7,650 stays), standardized under a single extraction and adjudication specification.
* **Hourly Physiological Grid**: 30 clinical variables placed on a shared 24-hour grid originating at Sepsis-3 onset.
* **Strict Quality Control**: Outliers violating strict biological bounds have been stripped from the raw temporal data (`hourly_observed.csv.gz`).
* **Deep Learning Imputation**: Includes a completely dense version of the time-series grid (`hourly_imputed.csv.gz`) reconstructed using a SAITS model trained exclusively on the MIMIC-IV development cohort to prevent data leakage.
* **Domain-Adapted Features**: Includes a summary representation (`features_transported.csv.gz`) aligned via entropic optimal transport for domain adaptation experiments.
* **Detailed Documentation**: Includes `cohort_flow.csv` and `data_dictionary.csv` detailing exact patient attrition at every step and exact per-variable missingness differences between the source databases.

---

## Acknowledgements

The author gratefully acknowledges the MIT Laboratory for Computational Physiology and the eICU Research Institute for compiling the MIMIC-IV and eICU-CRD databases, and PhysioNet for providing the credentialed access that made this work possible.

There is no funding, grant, or institutional support to declare for this dataset.

---

## Conflicts of interest

```
The author has no conflicts of interest to declare.
```

---

## Citation

When using this resource, please cite:

Kumar, P. (2026). *ClinicalTensorSepsis - Harmonised Sepsis-3 Cohorts with
Hourly Physiology Derived from MIMIC-IV-Ext and eICU-CRD* (version 1.0.0).
PhysioNet. RRID:SCR_007345.

The resource is currently under review. The permanent DOI will be added after
publication.

---

## License

The source code in this repository is released under the MIT License. See the
`LICENSE` file for details.

The MIT License applies to the source code only. The underlying credentialed
health data remain subject to the applicable PhysioNet license and data-use
agreement described above.

---

## References

1. Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10(1):1. \url{https://doi.org/10.1038/s41597-022-01899-x}.

2. Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Sci Data. 2018;5:180178. \url{https://doi.org/10.1038/sdata.2018.178}.

3. Singer M, Deutschman CS, Seymour CW, Shankar-Hari M, Annane D, Bauer M, et al. The third international consensus definitions for sepsis and septic shock (Sepsis-3). JAMA. 2016;315(8):801--10. \url{https://doi.org/10.1001/jama.2016.0287}.

4. Brown SM, Lanspa MJ, Jones JP, Kuttler KG, Li Y, Carlson R, et al. Survival after shock requiring high-dose vasopressor therapy. Chest. 2013;143(3):664--71. \url{https://doi.org/10.1378/chest.12-1106}.

5. Du W, Côté D, Liu Y. SAITS: self-attention-based imputation for time series. Expert Syst Appl. 2023;219:119619. \url{https://doi.org/10.1016/j.eswa.2023.119619}.

6. Courty N, Flamary R, Tuia D, Rakotomamonjy A. Optimal transport for domain adaptation. IEEE Trans Pattern Anal Mach Intell. 2017;39(9):1853--65. \url{https://doi.org/10.1109/TPAMI.2016.2615921}.

7. Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, et al. Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. Med Care. 2005;43(11):1130--9. \url{https://doi.org/10.1097/01.mlr.0000182534.19832.83}.

8. Kumar P. clinical_model_sepsis [Internet]. GitHub; 2026. Available from: \url{https://github.com/MrRajat1809/clinical_model_sepsis}.