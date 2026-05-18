from __future__ import annotations

from hestradiomics.config import CONFIG
from hestradiomics.hest import (
    run_download,
    run_gene_extraction,
)
from hestradiomics.segment.pipeline import (
    segment_all_oncotrees_from_config,
)
from hestradiomics.analysis import (
    run_visualization_from_config,
    save_overlays_all_oncotrees_from_config, 
)


def main():
    download_dir = CONFIG.download_dir
    sample_ids = CONFIG.sample_ids
    hest_config = CONFIG.hest
    segment_config = CONFIG.segment
    extract_config = CONFIG.extract
    visualize_config = CONFIG.visualize
    statstics_config = CONFIG.statistics

    # -------------------------------------------------------------------------
    # 1. Download HEST
    # -------------------------------------------------------------------------
    run_download(
        download_cfg=hest_config,
        sample_ids=sample_ids,
    )
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 2. Extract Gene Sets
    # -------------------------------------------------------------------------
    run_gene_extraction(
        download_cfg=hest_config,
        sample_ids=sample_ids,
        k_values=(50, 100, 250),
        criteria_values=("var", "mean"),
        min_cells_pct=0.1,
    )
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 3. Segment HEST Patches
    # -------------------------------------------------------------------------
    segment_paths = segment_all_oncotrees_from_config(
        download_cfg=hest_config,
        cellseg_cfg=segment_config,
        sample_ids=sample_ids,
    )

    print(f"[DONE] Segmented {len(segment_paths)} samples.")
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 4. Visualize Patches / Segment Overlays
    # -------------------------------------------------------------------------
    run_visualization_from_config(CONFIG)

    save_overlays_all_oncotrees_from_config(
        download_cfg=hest_config,
        cellseg_cfg=segment_config,
        sample_ids=sample_ids,
    )

    print("[DONE] Visualization completed.")
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 5. Extract Radiomics Features
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()