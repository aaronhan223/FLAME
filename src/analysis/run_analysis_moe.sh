cd /cis/home/schaud35/clinical-highmmt

python -m src.analysis.analyze_moe_weights \
    --ihm_model_path /cis/home/schaud35/clinical-highmmt/src/checkpoints/flame/multitask/ihm/ihm_TS-Text_mod_drop_rate_0.0.pt \
    --los_model_path /cis/home/schaud35/clinical-highmmt/src/checkpoints/flame/multitask/ihm/TS-Text/los_TS-Text-CXR_transfer_moe_from_ihm.pt \
    --output_dir /cis/home/schaud35/clinical-highmmt/src/analysis/analysis_results/moe_comparison
