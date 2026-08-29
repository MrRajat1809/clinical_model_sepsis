# PhysioNet submission text — ClinicalTensorSepsis

Draft for the PhysioNet project management system. Each section below maps to one
field in their submission form; paste them across one at a time.

Placeholders written as `{{LIKE_THIS}}` must be filled from the final pipeline
run. The value in brackets after each is what the previous run produced, given
only so you can sanity-check the new one — do not ship the old numbers.

Formatting rules this draft already follows, from `rules.txt`:

- no URLs anywhere in the body text; every external resource is a numbered
  reference cited as [n]
- Vancouver reference style
- title uses only letters, numbers, spaces and hyphens — no comma, no colon
- abstract is under 250 words and contains no references

---

## Project type

**Database.** Not Model or Software. The resource is a harmonised patient cohort
with hourly physiology; the summary feature matrix is a convenience derived from
it rather than the point of the release. No model weights are included, which
also keeps the submission clear of the requirement that machine-learning projects
be peer-reviewed before review.

---

## Title

```
ClinicalTensorSepsis - Harmonised Sepsis-3 Cohorts with Hourly Physiology Derived from MIMIC-IV and eICU-CRD
```

107 characters. If an editor reads "Derived from MIMIC-IV" as use of the MIMIC
acronym and asks for `Ext`, the fallback is:

```
MIMIC-IV-Ext-ClinicalTensorSepsis - Harmonised Sepsis-3 Cohorts with Hourly Physiology from MIMIC-IV and eICU-CRD
```

---

## Version

```
1.0.0
```

---

## Abstract

*(target under 250 words; current draft ~215)*

Sepsis is the leading cause of in-hospital mortality worldwide, and prognostic
models built on one intensive care database frequently degrade when applied to
another. Progress on that problem is limited by the absence of cohorts that are
constructed identically across sources, because differences in cohort definition
are otherwise indistinguishable from differences in physiology.

ClinicalTensorSepsis provides two Sepsis-3 cohorts assembled under a single
specification from MIMIC-IV and the eICU Collaborative Research Database.
{{TOTAL_N}} intensive care stays are included [previous run: 20,668], each with
thirty clinical variables placed on a common hourly grid spanning the first
twenty-four hours after sepsis onset. Sepsis onset is defined as the coincidence
of suspected infection with an acute increase of at least two points in the
Sequential Organ Failure Assessment score, adjudicated by the same code in both
databases.

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

## Methods and technical implementation

### Cohort construction

Both cohorts were restricted to adults aged at least eighteen years, retaining
the first intensive care stay per patient with a length of stay of at least
twenty-four hours, and excluding elective surgical admissions. In MIMIC-IV the
elective exclusion used the recorded admission type; in eICU-CRD it used the
elective surgery indicator in the APACHE predictor table. Because eICU-CRD does
not record absolute admission times, stays were ordered by ascending stay
identifier within each patient, and ages recorded as greater than eighty-nine
were assigned a value of ninety-one in accordance with the source de-identifi
cation scheme.

Suspected infection time was derived differently in the two databases, and this
is the single largest structural difference between them. In MIMIC-IV,
microbiological cultures were coupled to intravenous antibiotic administration
using the Sepsis-3 temporal rules: antibiotics beginning within seventy-two hours
after a culture, or a culture obtained within twenty-four hours after antibiotics
beginning. Antibiotic orders were first consolidated into continuous treatment
episodes so that consecutive courses and drug switches counted once. In eICU-CRD
the microbiology and medication tables are too sparsely populated for those rules
to be applied reliably, so suspected infection was taken from active and
admission diagnoses recorded within twenty-four hours of intensive care
admission.

Admissions whose primary or major diagnosis was a condition that reproduces
sepsis physiology without infection were excluded: acute myocardial infarction,
pulmonary embolism, acute pancreatitis, and trauma or burns. In eICU-CRD the
trauma criterion explicitly excludes organ-injury phrases, because the source
vocabulary describes acute kidney injury as an injury and an unrestricted match
would have removed septic acute kidney injury from that cohort alone.

### Sepsis-3 adjudication

The six-organ Sequential Organ Failure Assessment score was computed in two
windows: a baseline window from forty-eight hours before suspected infection up
to but not including that time, and an acute window from suspected infection to
twenty-four hours afterwards. Each component used the worst qualifying value in
its window. Stays whose score rose by at least two points were retained.

Sepsis onset was defined as the later of the suspected infection time and the
earliest acute-window observation crossing any organ-dysfunction threshold, so
that onset marks the point at which suspected infection and organ failure
coincide. Glasgow Coma Scale thresholds are component-specific because the
components have different maxima.

A component with no qualifying observation scores zero, and absent Glasgow Coma
Scale components are filled to normal before the total is formed. This follows
the usual operationalisation of Sepsis-3, but it interacts with the recording
density difference between the two databases and users should read it alongside
the per-variable density figures in the data dictionary.

### Hourly representation

Observations in the first twenty-four hours after sepsis onset were placed on a
common hourly grid of thirty variables, using the same feature order in both
databases. Aggregation within an hour follows the clinical meaning of each
variable rather than one blanket rule: temperature, sodium, potassium and
chloride are averaged; mean arterial pressure, oxygen saturation, platelets,
haemoglobin, albumin, pH and the Glasgow Coma Scale components take the minimum;
urine output is summed; all remaining variables and intervention indicators take
the maximum.

Two variables are engineered. The ratio of arterial oxygen tension to inspired
oxygen fraction is computed from harmonised measurements. Vasopressor doses are
converted to norepinephrine equivalents using published conversion factors [4],
computed only for hours in which a vasopressor was actually recorded so that no
drug remains distinct from zero dose.

Unit handling differs between the sources and was made explicit in both. In
MIMIC-IV every recorded infusion unit is mapped individually and unrecognised
units become missing rather than passing through unconverted. In eICU-CRD, where
dosing units are embedded in free-text drug descriptions, units are parsed from
the description, converted using recorded admission weight where available, and
records that cannot be converted are excluded and logged.

### Missing data

Values outside prespecified physiological bounds were set to missing rather than
clipped, so that implausible charting artefacts are reconstructed from the
patient's own trajectory rather than replaced by a boundary value.

Reconstruction used a self-attention imputation model for time series [5],
trained on the MIMIC-IV cohort alone. The eICU-CRD cohort was reconstructed by
applying the fitted scaler and weights without retraining, so that both cohorts
are reconstructed by the same function and any difference between them reflects
the data rather than two separately fitted models. After reconstruction the
transformations were reversed and values constrained to physiological ranges.

Both the observed and reconstructed versions are released. Users who wish to
apply their own imputation should use the observed file, in which a blank cell
denotes no observation in that hour.

### Cross-database alignment

A summary representation was formed by taking the mean, minimum, maximum and
standard deviation of each of the thirty variables across the twenty-four hour
window, giving one hundred and twenty features, to which two static variables,
age and baseline SOFA, were appended.

An additional file provides that representation after entropic Sinkhorn optimal
transport [6], fitted with eICU-CRD as source and MIMIC-IV as target, so that the
eICU distribution is mapped onto the MIMIC-IV one. Values in the eICU rows of
that file are mapped rather than measured and must not be interpreted as
observations. It is provided because alignment of this kind is a common step in
domain-adaptation work and reproducing it exactly is otherwise laborious.

### Comorbidity

The Charlson Comorbidity Index was computed in both databases from coded
diagnoses using the enhanced ICD-9-CM and ICD-10 algorithms of Quan and
colleagues [7]. Because eICU-CRD records a substantial share of diagnoses as free
text with no code, a restricted set of diagnosis-string terms was used for
entries carrying no usable code; entries with a code were scored from the code
alone in both databases.

The index is provided in the cohort table but users should read the limitation
recorded in Usage Notes before treating it as a measure of pre-admission burden.

{{CCI_LIMITATION_WORDING}}

---

## Content description

All files are gzip-compressed comma-separated values, formatted per RFC 4180,
except the data dictionary and cohort flow, which are uncompressed.

| File | Rows | Description |
| --- | --- | --- |
| `readme.txt` | — | package description and file summary |
| `cohort.csv.gz` | {{TOTAL_N}} | one row per intensive care stay |
| `hourly_observed.csv.gz` | {{TOTAL_N}} x 24 | measured values; blank means no observation |
| `hourly_imputed.csv.gz` | {{TOTAL_N}} x 24 | reconstructed values; complete |
| `features_model.csv.gz` | {{TOTAL_N}} | summary representation, 122 columns |
| `features_transported.csv.gz` | {{TOTAL_N}} | same representation after optimal transport |
| `data_dictionary.csv` | ~160 | per-variable documentation |
| `cohort_flow.csv` | ~15 | cohort attrition at each step |
| `sha256sums.txt` | — | checksums for every file |

Cohort sizes are {{MIMIC_N}} stays from MIMIC-IV [previous run: 13,018] and
{{EICU_N}} from eICU-CRD [previous run: 7,650].

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

A manuscript describing an analysis built on this resource is in preparation.

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

{{CCI_LIMITATION_WORDING}}

### Complementary resources

The complete code that constructs this resource, including the exact queries and
the export script, is available [8]. The source databases are [1] and [2].

---

## Ethics

This work used two publicly available de-identified databases, MIMIC-IV [1] and
the eICU Collaborative Research Database [2], both distributed through PhysioNet
under credentialed access. The author completed the required human subjects
research training and is party to the applicable data use agreements.

No additional institutional review board approval was sought, because the work
involves no new data collection, no intervention, and no contact with patients,
and both source databases carry their own approvals and waivers of informed
consent for research use.

No attempt was made to identify any individual. No information beyond that
already present in the source databases has been added. All times are expressed
as offsets rather than dates, and no absolute dates appear in any released file.

The intended benefit is to make cross-database evaluation of clinical prediction
models easier and more honest, by removing differences in cohort definition as a
competing explanation for a model failing to transport. The principal risk is
misuse of the released representations as though they were raw measurements,
particularly the transported file, in which the eICU values are mapped rather
than observed. That risk is addressed by separating observed from reconstructed
from transported into distinct files and by documenting the distinction in the
data dictionary, the readme and Usage Notes.

The resource is derived from credentialed sources and is released under the same
credentialed terms, so access remains restricted to users who have completed the
required training and agreed to the data use agreement.

---

## Acknowledgements

{{ACKNOWLEDGEMENTS}}

Suggested content: the teams responsible for MIMIC-IV and eICU-CRD and for
maintaining PhysioNet; any institutional support; and funding, with grant numbers
if applicable. If there is no funding to declare, say so explicitly.

---

## Conflicts of interest

```
The author has no conflicts of interest to declare.
```

---

## References

Vancouver style, numbered in order of first citation in the body text.

1. Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S, et al.
   MIMIC-IV a freely accessible electronic health record dataset. Sci Data. 2023;
   10:1. {{VERIFY_CITATION}}

2. Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU
   Collaborative Research Database a freely available multi-center database for
   critical care research. Sci Data. 2018; 5:180178. {{VERIFY_CITATION}}

3. Singer M, Deutschman CS, Seymour CW, Shankar-Hari M, Annane D, Bauer M, et al.
   The Third International Consensus Definitions for Sepsis and Septic Shock
   Sepsis-3. JAMA. 2016; 315:801-810. {{VERIFY_CITATION}}

4. Brown SM, Lanspa MJ, Jones JP, Kuttler KG, Li Y, Carlson R, et al. Survival
   after shock requiring high-dose vasopressor therapy. Chest. 2013;
   143:664-671. {{VERIFY_CITATION}}

5. Du W, Cote D, Liu Y. SAITS self-attention-based imputation for time series.
   Expert Syst Appl. 2023; 219:119619. {{VERIFY_CITATION}}

6. Courty N, Flamary R, Tuia D, Rakotomamonjy A. Optimal transport for domain
   adaptation. IEEE Trans Pattern Anal Mach Intell. 2017; 39:1853-1865.
   {{VERIFY_CITATION}}

7. Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, et al. Coding
   algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative
   data. Med Care. 2005; 43:1130-1139. {{VERIFY_CITATION}}

8. {{GITHUB_REPOSITORY_CITATION}} — the code repository. Cite as a software
   reference with author, title, version or commit, and year. The URL belongs
   here, not in the body text.

Every `{{VERIFY_CITATION}}` marks a reference I have written from memory. Check
each against the actual record before submitting; volume, page and year in
particular.

---

## Pre-submission checklist

- [ ] fill every `{{PLACEHOLDER}}` from the final run
- [ ] verify all eight references against the real records
- [ ] confirm no absolute dates in any released file
- [ ] confirm no `.joblib`, `.pkl` or other pickle-format file is included
- [ ] confirm no URL appears in any body text section
- [ ] run `physionet validate release/`
- [ ] run `croissant-baker --input release/`
- [ ] regenerate `sha256sums.txt` after any file changes
- [ ] select licence: PhysioNet Credentialed Health Data License 1.5.0
- [ ] select project type: Database
- [ ] confirm cohort counts in the abstract match `cohort_flow.csv`
