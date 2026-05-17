from typing import List 

from hestradiomics.config import CONFIG

from hestradiomics._download_hest import (
    run_download, 
    run_gene_extraction,
)
from hestradiomics._segment import (
    verify_or_download_model, 
    segment_all_oncotrees_from_config, 
)
from hestradiomics._extract import (
    run_radiomics_extraction_from_config, 
)


def run():
    cfg = CONFIG

    ### 

    if cfg.run.run_hest_download:
        run_download(
            download_cfg=cfg.download,
            sample_ids=cfg.sample_ids,
        )
        run_gene_extraction(
            download_cfg=cfg.download,
            sample_ids=cfg.sample_ids,
        )
    
    print(f"[INFO] download done.")

    ### 

    verify_or_download_model(
        model_path=str(cfg.cellseg.model_path),
        model_name=cfg.cellseg.model_name,
    )

    segment_paths: List[str] = []

    if cfg.run.run_segment:
        segment_paths = segment_all_oncotrees_from_config(
            download_cfg=cfg.download,
            cellseg_cfg=cfg.cellseg,
            sample_ids=cfg.sample_ids,
        )

    print(f"[INFO] segment done.")
    print(f"[INFO] segmented samples: {len(segment_paths)}")

    ###
    if cfg.run.run_radiomics_extraction:
        run_radiomics_extraction_from_config(config=CONFIG)

    print(f"[INFO] extract done.")

    ### 

    # if cfg.run.run_statistics:
    #     run_statistics(cfg.download, cfg.statistics)

    print("[INFO] pipeline finished successfully")


if __name__ == "__main__":
    run()



