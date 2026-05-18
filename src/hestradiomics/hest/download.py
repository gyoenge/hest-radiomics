from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import (
    HfApi,
    login,
    snapshot_download,
)

from hestradiomics.config import DownloadConfig
from hestradiomics.utils import (
    ensure_dir,
    filter_sample_ids,
)


def huggingface_checkin() -> None:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. "
            "Set HF_TOKEN."
        )

    login(token=hf_token)

    try:
        user_info = HfApi().whoami()

        print(
            f"Current Hugging Face Account: "
            f"{user_info['name']}"
        )

    except Exception as e:
        print(
            f"Failed to check Hugging Face account: {e}"
        )


def build_hest_allow_patterns(
    sample_ids: Sequence[str],
    required_dirs: Sequence[str],
    optional_dirs: Sequence[str],
) -> list[str]:
    target_dirs = list(
        dict.fromkeys(
            [*required_dirs, *optional_dirs]
        )
    )

    allow_patterns = []

    for dirname in target_dirs:
        for sample_id in sample_ids:

            allow_patterns.append(
                f"{dirname}/*{sample_id}[_.]**"
            )

    return allow_patterns


def download_hest_by_oncotree(
    download_cfg: DownloadConfig,
    sample_ids: Optional[tuple[str, ...]] = None,
    metadata_uri: str = (
        "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv"
    ),
) -> None:
    download_dir = download_cfg.download_dir

    print(f"Loading HEST metadata: {metadata_uri}")

    meta_df = pd.read_csv(metadata_uri)

    meta_df = meta_df[
        (meta_df["species"] == "Homo sapiens")
        & (
            meta_df["oncotree_code"]
            .isin(download_cfg.oncotrees)
        )
        & (
            meta_df["st_technology"]
            .isin(download_cfg.technologies)
        )
    ]

    for oncotree_code in download_cfg.oncotrees:

        oncotree_df = meta_df[
            meta_df["oncotree_code"]
            == oncotree_code
        ]

        if oncotree_df.empty:
            print(
                f"[SKIP] No samples found for "
                f"{oncotree_code}"
            )
            continue

        all_sample_ids = (
            oncotree_df["id"]
            .astype(str)
            .tolist()
        )

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        if not target_sample_ids:
            print(
                f"[SKIP] No selected samples found "
                f"for {oncotree_code}"
            )
            continue

        allow_patterns = build_hest_allow_patterns(
            sample_ids=target_sample_ids,
            required_dirs=download_cfg.required_dirs,
            optional_dirs=download_cfg.optional_dirs,
        )

        oncotree_dir = ensure_dir(
            download_dir / oncotree_code
        )

        print(
            f"\nStart downloading HEST | "
            f"oncotree={oncotree_code}"
        )

        print(f"download_dir: {oncotree_dir}")

        print(
            f"num_samples: "
            f"{len(target_sample_ids)} / "
            f"{len(all_sample_ids)}"
        )

        snapshot_download(
            repo_id="MahmoodLab/hest",
            repo_type="dataset",
            local_dir=str(oncotree_dir),
            allow_patterns=allow_patterns,
        )

        print(
            f"Completed downloading HEST | "
            f"oncotree={oncotree_code}"
        )


def run_download(
    download_cfg: DownloadConfig,
    sample_ids: Optional[tuple[str, ...]] = None,
) -> None:
    huggingface_checkin()

    download_hest_by_oncotree(
        download_cfg=download_cfg,
        sample_ids=sample_ids,
    )

    print("HEST download completed")

