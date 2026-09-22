The registry data are not included in this repository; see README.md.

Two files belong here if you hold them under a data use agreement:

  final_df.csv                 item-level export, read by 01, 02 and 02b
  imputed_clinical_data.csv    analytic file, written by 01 and read by 03,
                               06 and 07

synthetic/ holds a stand-in built from published aggregates alone. It has the
same columns as imputed_clinical_data.csv, so the three scripts that read that
file also run against it:

  python code/03_phase2_analyses.py --data data/synthetic/synthetic_clinical_data.csv

Scripts 01, 02 and 02b need the item-level export and cannot run against it.
Script 05 needs no data at all and reproduces Study 1 exactly.
