"""Build the canonical corpus/tokenizer provenance artifact from local evidence."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.utils.artifacts import atomic_write_json, sha256_file  # noqa: E402
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402

DATA_CONFIG = Path("configs/data/fineweb_edu_10bt.yaml")
TOKENIZER_CONFIG = Path("configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml")
OUTPUT = Path("results/analysis/final_data_tokenizer_provenance.json")


def _artifact_record(path_value: str) -> dict[str, int | str]:
    path = PROJECT_ROOT / path_value
    if not path.is_file():
        raise FileNotFoundError(f"Missing required provenance artifact: {path}")
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    data = load_yaml_config(PROJECT_ROOT / DATA_CONFIG)
    tokenizer = load_yaml_config(PROJECT_ROOT / TOKENIZER_CONFIG)
    artifact_paths = sorted(
        {
            *data["artifacts"].values(),
            *tokenizer["artifacts"].values(),
        }
    )
    artifacts = {path: _artifact_record(path) for path in artifact_paths}

    corpus_metadata = load_yaml_config(PROJECT_ROOT / data["artifacts"]["metadata"])
    tokenized_metadata = load_yaml_config(PROJECT_ROOT / data["artifacts"]["tokenized_metadata"])
    tokenizer_metadata = load_yaml_config(PROJECT_ROOT / tokenizer["artifacts"]["metadata_json"])

    payload = {
        "artifact_type": "final_data_tokenizer_provenance",
        "schema_version": 2,
        "study_scope": ("Balanced primary matrix: 6 models x 3 token budgets x 10 aligned seeds."),
        "data_config": DATA_CONFIG.as_posix(),
        "data_config_sha256": sha256_file(PROJECT_ROOT / DATA_CONFIG),
        "tokenizer_config": TOKENIZER_CONFIG.as_posix(),
        "tokenizer_config_sha256": sha256_file(PROJECT_ROOT / TOKENIZER_CONFIG),
        "dataset": data["dataset"],
        "streaming": data["streaming"],
        "preprocessing": data["preprocessing"],
        "split_caps": data["caps"],
        "tokenizer": {
            **tokenizer["tokenizer"],
            "configured_vocab_size": tokenizer["tokenizer"]["vocab_size"],
            "special_tokens": tokenizer["special_tokens"],
            "max_training_chars": tokenizer["training"]["max_training_chars"],
            "actual_vocab_size": tokenizer_metadata["actual_vocab_size"],
            "effective_training_chars": tokenizer_metadata["training"]["effective_training_chars"],
            "was_capped": tokenizer_metadata["training"]["was_capped"],
            "byte_alphabet_coverage": tokenizer_metadata["byte_alphabet_coverage"],
        },
        "corpus": corpus_metadata,
        "tokenization": tokenized_metadata,
        "artifacts": artifacts,
        "regeneration_commands": [
            "python scripts/data/prepare_hf_streaming_text.py --config " + DATA_CONFIG.as_posix(),
            "python scripts/tokenizer/train_tokenizer.py --config " + TOKENIZER_CONFIG.as_posix(),
            "python scripts/data/tokenize_lm_corpus.py --data-config "
            + DATA_CONFIG.as_posix()
            + " --tokenizer-config "
            + TOKENIZER_CONFIG.as_posix(),
        ],
    }
    atomic_write_json(PROJECT_ROOT / OUTPUT, payload)
    print(f"Wrote {OUTPUT} with {len(artifacts)} verified artifacts")


if __name__ == "__main__":
    main()
