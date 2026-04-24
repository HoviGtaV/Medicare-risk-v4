from pathlib import Path
import pandas as pd

OUT_DIR = Path("submission/qa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

full_checklist = [
    "README.md",
    "TEAM_CONTRIBUTIONS.md",
    "Dockerfile",
    "requirements.txt",
    "score_batch_v4.py",
    "examples/sample_input_v4.csv",
    "examples/sample_output_v4.csv",
    "models_v4/final/model.cbm",
    "models_v4/final/metrics_test.json",
    "models_v4/final/feature_importance.csv",
    "outputs/metrics_v4/model_comparison_v4.csv",
    "outputs/shap_v4/shap_summary_bar.png",
    "outputs/shap_v4/shap_beeswarm.png",
    "outputs/shap_v4/global_shap_importance.csv",
    "outputs/shap_v4/local_explanations_top5.csv",
    "submission/docs/LABEL_POLICY_APPENDIX.md",
    "submission/docs/PIPELINE_TRACE.md",
    "submission/docs/data_dictionary_v4_full.csv",
    "submission/docs/DATA_DICTIONARY_V4_FULL.md",
    "submission/monitoring/score_drift_summary.csv",
    "submission/monitoring/feature_drift_summary.csv",
    "submission/monitoring/score_distribution_2022_vs_2023.png",
    "submission/monitoring/MONITORING_NOTE.md",
    "submission/report/report_one_page_final.md",
    "submission/slides/presentation_final_content.md",
    "submission/video/video_demo_script_final.md",
    "submission/links/DOCKERHUB_URL.txt",
    "submission/links/GITHUB_REPO_URL.txt",
    "submission/qa/check_final_feature_tables_v4.csv",
    "submission/qa/check_final_labels_v4.csv",
    "submission/qa/check_oig_date_consistency_v4.csv",
    "submission/qa/final_active_manifest.csv",
]

hard_deliverables = [
    "score_batch_v4.py",
    "models_v4/final/model.cbm",
    "models_v4/final/metrics_test.json",
    "submission/docs/DATA_DICTIONARY_V4_FULL.md",
    "submission/docs/PIPELINE_TRACE.md",
    "submission/report/report_one_page_final.md",
    "submission/slides/presentation_final_content.md",
    "submission/video/video_demo_script_final.md",
    "submission/monitoring/score_drift_summary.csv",
    "submission/monitoring/feature_drift_summary.csv",
]

checklist_df = pd.DataFrame({"file": full_checklist})
checklist_df["exists"] = checklist_df["file"].apply(lambda x: Path(x).exists())

hard_df = pd.DataFrame({"file": hard_deliverables})
hard_df["exists"] = hard_df["file"].apply(lambda x: Path(x).exists())

checklist_path = OUT_DIR / "final_submission_checklist.csv"
hard_path = OUT_DIR / "hard_deliverables_check.csv"

checklist_df.to_csv(checklist_path, index=False)
hard_df.to_csv(hard_path, index=False)

print(checklist_df.to_string(index=False))
print("\nSaved:", checklist_path)
print("Saved:", hard_path)
