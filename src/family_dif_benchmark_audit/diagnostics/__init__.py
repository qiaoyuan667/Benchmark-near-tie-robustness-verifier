"""Post-hoc diagnostics using the frozen owner-disjoint primary audit."""


def require_primary_files(primary_root, keys, reconstruct=False):
    """Fail with actionable relative filenames before creating diagnostic outputs."""
    names = ["owner_family_representatives.csv", "crossfit_anchor_items.csv",
             "matched_random_anchor_controls.csv", "ranking_impact_metrics.csv"]
    if reconstruct:
        names += ["family_models.csv", "dimension_selection.csv", "crossfit_model_coordinates.csv"]
    missing = [str((key + "_family_ranking_impact") + "/" + name)
               for key in keys for name in names
               if not (primary_root / (key + "_family_ranking_impact") / name).is_file()]
    if missing:
        raise FileNotFoundError("Reconstruct primary outputs first or set --primary-root. Missing: "
                                + ", ".join(missing))
