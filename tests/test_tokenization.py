from __future__ import annotations

import numpy as np
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from deepseek_reimpl.data.tokenization import encode_text, encode_text_file_to_int32_bin
from deepseek_reimpl.tokenizer.train_tokenizer import train_byte_level_bpe_tokenizer
from tokenizers import Tokenizer


def _tokenizer_with_bos_eos_post_processor() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "alpha": 3,
                "beta": 4,
                "gamma": 5,
                "delta": 6,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    return tokenizer


def test_lm_text_encoding_disables_tokenizer_special_token_injection(tmp_path) -> None:
    tokenizer = _tokenizer_with_bos_eos_post_processor()
    text = "alpha beta\ngamma delta\n"
    input_path = tmp_path / "corpus.txt"
    output_path = tmp_path / "tokens.int32.bin"
    input_path.write_text(text, encoding="utf-8")

    num_tokens = encode_text_file_to_int32_bin(
        input_path,
        tokenizer,
        output_path,
        batch_lines=1,
    )
    token_ids = np.fromfile(output_path, dtype=np.int32).tolist()

    assert token_ids == encode_text(text, tokenizer)
    assert num_tokens == len(token_ids)
    assert tokenizer.token_to_id("<bos>") not in token_ids
    assert tokenizer.token_to_id("<eos>") not in token_ids


def test_incremental_lm_encoding_is_invariant_to_line_batch_size(tmp_path) -> None:
    tokenizer = _tokenizer_with_bos_eos_post_processor()
    input_path = tmp_path / "corpus.txt"
    output_one = tmp_path / "tokens-one.int32.bin"
    output_many = tmp_path / "tokens-many.int32.bin"
    input_path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    encode_text_file_to_int32_bin(input_path, tokenizer, output_one, batch_lines=1)
    encode_text_file_to_int32_bin(input_path, tokenizer, output_many, batch_lines=4)

    assert output_one.read_bytes() == output_many.read_bytes()


def test_real_byte_level_incremental_encoding_matches_one_shot(tmp_path) -> None:
    text = "alpha beta\n\nUnicode: café 雪\nline three\n"
    input_path = tmp_path / "corpus.txt"
    output_path = tmp_path / "tokens.int32.bin"
    input_path.write_text(text * 8, encoding="utf-8")
    tokenizer, _, _ = train_byte_level_bpe_tokenizer(
        input_text_files=[input_path],
        vocab_size=512,
        min_frequency=1,
        special_tokens_config={
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
        },
    )

    encode_text_file_to_int32_bin(
        input_path,
        tokenizer,
        output_path,
        batch_lines=1,
    )

    assert np.fromfile(output_path, dtype=np.int32).tolist() == encode_text(
        input_path.read_text(encoding="utf-8"), tokenizer
    )
