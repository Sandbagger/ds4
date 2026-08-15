#!/usr/bin/env python3
"""Focused source-contract tests for the Laguna Poolside oracle helper."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GGUF_TOOLS = ROOT / "gguf-tools"
DUMPER = HERE / "dump_llama_logits.cpp"
CASES = ROOT / "tests/test-vectors/laguna-resident/cases.json"
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"


class DumpLlamaContractTest(unittest.TestCase):
    def test_new_capture_pin_excludes_the_historical_artifact(self) -> None:
        source = DUMPER.read_text(encoding="utf-8")
        self.assertIn("e2ccc0579fc18e6ea2362fa25fccbcd470f0e332", source)
        self.assertIn("68248760064", source)
        self.assertIn(
            "a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff",
            source,
        )
        self.assertNotIn("706fa69799926b6afde1af9e24ca2a4923f110a1", source)
        self.assertNotIn(
            "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
            source,
        )

    def test_checked_in_cases_use_exact_ctx_schema(self) -> None:
        self.assertEqual(
            json.loads(CASES.read_text(encoding="utf-8")),
            {
                "schema": "laguna-resident-oracle-v1",
                "vocab_size": 100352,
                "continuation_case": "yarn-8193",
                "continuation_tokens": 8,
                "cases": [
                    {
                        "id": "short",
                        "render": "laguna-ds4",
                        "prompt": "short.txt",
                        "ctx": 1024,
                    },
                    {
                        "id": "swa-513",
                        "render": "raw",
                        "prompt": "swa-513.prompt",
                        "frontier": 513,
                        "ctx": 1024,
                    },
                    {
                        "id": "yarn-8193",
                        "render": "raw",
                        "prompt": "yarn-8193.prompt",
                        "frontier": 8193,
                        "ctx": 8202,
                    },
                    {
                        "id": "deep-32768",
                        "render": "raw",
                        "prompt": "deep-32768.prompt",
                        "frontier": 32768,
                        "ctx": 32768,
                    },
                ],
            },
        )

    def test_capture_v2_records_template_bytes_and_explicit_renderings(self) -> None:
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            include = root / "include"
            include.mkdir()
            (include / "ggml-backend.h").write_text(
                "#pragma once\ninline void ggml_backend_load_all() {}\n",
                encoding="utf-8",
            )
            (include / "llama.h").write_text(
                textwrap.dedent(
                    """\
                    #pragma once
                    #include <cstdint>

                    using llama_token = int32_t;
                    using llama_pos = int32_t;
                    using llama_seq_id = int32_t;
                    struct llama_vocab {};
                    struct llama_model {};
                    struct llama_context {};
                    struct llama_batch {
                        int32_t n_tokens = 0;
                        llama_token *token = nullptr;
                        llama_pos *pos = nullptr;
                        int32_t *n_seq_id = nullptr;
                        llama_seq_id **seq_id = nullptr;
                        int8_t *logits = nullptr;
                    };
                    struct llama_context_params {
                        uint32_t n_ctx = 0;
                        uint32_t n_batch = 0;
                        uint32_t n_ubatch = 0;
                        uint32_t n_seq_max = 0;
                        bool no_perf = false;
                    };
                    struct llama_model_params {
                        int32_t n_gpu_layers = 0;
                        bool use_mmap = false;
                    };

                    inline int32_t llama_tokenize(
                        const llama_vocab *, const char *, int32_t,
                        llama_token *, int32_t, bool, bool) { return 0; }
                    inline int32_t llama_detokenize(
                        const llama_vocab *, const llama_token *, int32_t,
                        char *, int32_t, bool, bool) { return 0; }
                    inline void llama_model_free(llama_model *) {}
                    inline void llama_free(llama_context *) {}
                    inline llama_batch llama_batch_init(int32_t, int32_t, int32_t) { return {}; }
                    inline void llama_batch_free(llama_batch) {}
                    inline int llama_decode(llama_context *, llama_batch) { return 0; }
                    inline const float *llama_get_logits_ith(llama_context *, int32_t) { return nullptr; }
                    inline llama_context_params llama_context_default_params() { return {}; }
                    inline llama_context *llama_init_from_model(
                        llama_model *, llama_context_params) { return nullptr; }
                    inline uint32_t llama_n_batch(const llama_context *) { return 0; }
                    inline void llama_backend_init() {}
                    inline void llama_backend_free() {}
                    inline llama_model_params llama_model_default_params() { return {}; }
                    inline llama_model *llama_model_load_from_file(
                        const char *, llama_model_params) { return nullptr; }
                    inline const llama_vocab *llama_model_get_vocab(const llama_model *) { return nullptr; }
                    inline int32_t llama_vocab_n_tokens(const llama_vocab *) { return 0; }
                    inline const char *llama_model_chat_template(
                        const llama_model *, const char *) { return nullptr; }
                    """
                ),
                encoding="utf-8",
            )

            harness = root / "capture_harness.cpp"
            harness.write_text(
                textwrap.dedent(
                    f"""\
                    #define main dump_llama_program_main
                    #include "{DUMPER.as_posix()}"
                    #undef main

                    int main(int argc, char **argv) {{
                        if (argc != 2) return 2;
                        fs::path output = argv[1];
                        fs::create_directories(output);
                        DumpResult yarn;
                        yarn.test_case.id = "yarn-8193";
                        yarn.test_case.render = "raw";
                        yarn.test_case.prompt_name = "yarn-8193.prompt";
                        yarn.test_case.frontier = 8193;
                        yarn.test_case.context = 8202;
                        yarn.continuation_argmax.assign(8, 7);
                        std::vector<DumpResult> results = {{yarn}};
                        const std::string chat_template =
                            "{{% if enable_thinking %}}<think>"
                            "{{% else %}}</think>{{% endif %}}";
                        const std::string probe = "template probe\\n";
                        write_capture(
                            output, 100352, "yarn-8193", 8, 40001, results,
                            chat_template, probe);
                        return 0;
                    }}
                    """
                ),
                encoding="utf-8",
            )
            binary = root / "capture-harness"
            compiled = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O0",
                    "-I",
                    str(include),
                    str(harness),
                    "-o",
                    str(binary),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)

            output = root / "capture"
            subprocess.run([str(binary), str(output)], check=True)
            capture = json.loads((output / "capture.json").read_text(encoding="utf-8"))
            template = b"{% if enable_thinking %}<think>{% else %}</think>{% endif %}"
            prefix = (
                "\u3008|EOS|\u3009<system>Laguna template reconciliation probe."
                "</system>\n<user>template probe\n</user>\n<assistant>"
            ).encode()
            think = prefix + b"<think>"
            nothink = prefix + b"</think>"
            self.assertEqual(capture["schema"], "laguna-resident-capture-v2")
            self.assertEqual(capture["seed_token_count"], 40001)
            self.assertEqual(
                capture["model"],
                {
                    "repository": "poolside/Laguna-S-2.1-GGUF",
                    "revision": "e2ccc0579fc18e6ea2362fa25fccbcd470f0e332",
                    "file": "laguna-s-2.1-Q4_K_M.gguf",
                    "size": 68248760064,
                    "sha256": "a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff",
                },
            )
            self.assertEqual(
                capture["chat_template"],
                {
                    "file": "tokenizer.chat_template.jinja",
                    "bytes": len(template),
                    "sha256": hashlib.sha256(template).hexdigest(),
                    "render_contract": "pinned-template-semantics-v1",
                    "reconciliation_system": "Laguna template reconciliation probe.",
                    "think_prompt_file": "chat-template-think.prompt",
                    "think_prompt_sha256": hashlib.sha256(think).hexdigest(),
                    "nothink_prompt_file": "chat-template-nothink.prompt",
                    "nothink_prompt_sha256": hashlib.sha256(nothink).hexdigest(),
                },
            )
            self.assertEqual((output / "tokenizer.chat_template.jinja").read_bytes(), template)
            self.assertEqual((output / "chat-template-think.prompt").read_bytes(), think)
            self.assertEqual((output / "chat-template-nothink.prompt").read_bytes(), nothink)
            for name in (
                "tokenizer.chat_template.jinja",
                "chat-template-think.prompt",
                "chat-template-nothink.prompt",
            ):
                self.assertEqual(
                    capture["files"][name],
                    hashlib.sha256((output / name).read_bytes()).hexdigest(),
                )

    def test_make_target_rejects_wrong_or_dirty_llama_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            checkout = root / "llama.cpp"
            bin_dir.mkdir()
            checkout.mkdir()
            fake_git = bin_dir / "git"
            fake_git.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    if [ "$1" = "-C" ]; then
                        shift 2
                    fi
                    case "$1" in
                        rev-parse)
                            printf '%s\n' "$FAKE_GIT_HEAD"
                            ;;
                        status)
                            if [ "${FAKE_GIT_DIRTY:-0}" = "1" ]; then
                                printf '%s\n' ' M include/llama.h'
                            elif [ "${FAKE_GIT_UNTRACKED:-0}" = "1" ]; then
                                case " $* " in
                                    *" --untracked-files=no "*) ;;
                                    *) printf '%s\n' '?? build/' ;;
                                esac
                            fi
                            ;;
                        *)
                            exit 2
                            ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            fake_git.chmod(0o755)

            def run_make(
                head: str,
                dirty: bool = False,
                untracked: bool = False,
                extra_args: tuple[str, ...] = (),
            ) -> subprocess.CompletedProcess[str]:
                env = os.environ.copy()
                env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
                env["FAKE_GIT_HEAD"] = head
                env["FAKE_GIT_DIRTY"] = "1" if dirty else "0"
                env["FAKE_GIT_UNTRACKED"] = "1" if untracked else "0"
                return subprocess.run(
                    [
                        "make",
                        "-C",
                        str(GGUF_TOOLS),
                        "quality-laguna-logits",
                        f"LLAMA_CPP_DIR={checkout}",
                        "CXX=true",
                        "LLAMA_CPP_CXXFLAGS=",
                        "LLAMA_CPP_LDLIBS=",
                        *extra_args,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )

            wrong = run_make("0" * 40)
            self.assertNotEqual(wrong.returncode, 0)
            self.assertIn(LLAMA_COMMIT, wrong.stdout + wrong.stderr)

            override_attempt = run_make(
                "0" * 40,
                extra_args=(f"LLAMA_LAGUNA_COMMIT={'0' * 40}",),
            )
            self.assertNotEqual(override_attempt.returncode, 0)
            self.assertIn(LLAMA_COMMIT, override_attempt.stdout + override_attempt.stderr)

            dirty = run_make(LLAMA_COMMIT, dirty=True)
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn("tracked checkout is dirty", dirty.stdout + dirty.stderr)

            clean = run_make(LLAMA_COMMIT)
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)

            untracked = run_make(LLAMA_COMMIT, untracked=True)
            self.assertEqual(untracked.returncode, 0, untracked.stdout + untracked.stderr)


if __name__ == "__main__":
    unittest.main()
