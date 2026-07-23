LABELS = ["S_mean", "synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
ID_COLUMNS = ["drug_row", "drug_col", "depmap"]

METHOD_NAME = "GAR-Syn"

RAW_FILES = [
    "Drug_map.npy",
    "drug_feature_graph.npy",
    "nv_zscore.csv",
    "mutation.csv",
    "Drug_use.csv",
    "Cell_use_zscore.csv",
    "drug_sequence_em.csv",
    "data_to_split.csv",
]
