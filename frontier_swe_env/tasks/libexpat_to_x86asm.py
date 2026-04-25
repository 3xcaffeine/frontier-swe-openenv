"""libexpat-to-x86asm task configuration.

Agent reimplements libexpat 2.6.4 in x86-64 assembly and produces a shared
library at ``/app/asm-port/libexpat.so`` that exports the libexpat C ABI.

L1 scoring reads the upstream verifier's structured reward.json. The
verifier (``/opt/verifier/test.sh``) decrypts gcc, builds a reference C
libexpat .so, compiles the upstream test suite + benchmarks, runs them
against the agent's .so, then ``compute_reward.py`` writes a numeric
``score`` field combining correctness (50%) and performance (50%, ratio
of ref_time / agent_time across small/medium/large XML docs). Hard-fails
(no .so, anti-cheat trip, gcc-toolchain failure, zero correctness) emit
``score == 0.0``.
"""

from __future__ import annotations

from pathlib import Path

from ..task_config import TaskConfig


LIBEXPAT_TRAINING_INSTRUCTION = """
# libexpat to x86-64 Assembly

Your workspace is `/app/asm-port/`. The full libexpat 2.6.4 C source ships at
`/app/expat-src/lib/` (read-only reference). You must reimplement libexpat in
**x86-64 assembly** and produce a shared library:

```
/app/asm-port/libexpat.so
```

This `.so` must export the libexpat C ABI (functions declared in
`/app/expat-src/lib/expat.h`) so C programs compiled against `expat.h` link
and call your library directly.

## Available tools

`nasm`, `as`, `ld`, `objdump`, `readelf`, `nm`, `strace`, `make`, `libc6-dev`.

**There is no C compiler.** You can call libc functions (`malloc`, `free`,
`memcpy`, etc.) from assembly — `libc6-dev` is installed for linking.

## Hard rules (verifier anti-cheat)

The verifier rejects (hard-fail, score = 0.0) any submission that:

1. Doesn't produce a `.so` exporting `XML_ParserCreate` in `/app/asm-port/`.
2. Has zero `.s` or `.asm` source files in `/app/asm-port/`.
3. Includes any C source named `xmlparse.c`, `xmltok.c`, `xmlrole.c`,
   `xmltok_impl.c`, or `xmltok_ns.c` in `/app/asm-port/`.
4. References `dlopen`, `dlsym`, or `RTLD_` in your assembly source.
5. Has a `NEEDED libexpat` dynamic dependency in the produced `.so`.
6. Embeds system libexpat path strings (e.g. `/usr/lib/.../libexpat...`).
7. Has a `.comment` section showing it was compiled by GCC or clang.

## ABI

System V AMD64: args in `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`; return in
`rax`; callee-saved `rbx`, `rbp`, `r12`-`r15`.

## Scoring

The verifier (`bash /opt/verifier/test.sh`) runs in stages:

1. **Find agent .so** — looks for a file in `/app/asm-port/` whose dynamic
   symbol table exports `T XML_ParserCreate`.
2. **Anti-cheat** — see rules above.
3. **Reference build** — verifier decrypts gcc and builds a reference C
   libexpat as a baseline.
4. **Test suite** — links the upstream expat test suite (basic, ns, misc,
   alloc, nsalloc) against your `.so` and runs `runtests`. Per-module pass
   rates are weighted (basic=3, ns=2, misc=1, alloc=2, nsalloc=1).
5. **Benchmarks** — three XML docs (small/medium/large) parsed by the
   reference and your library; ratio `ref_time / agent_time` per doc with
   weights small=1, medium=1, large=2.
6. **Reward** — `0.5 * correctness + 0.5 * performance` when correctness > 0;
   else 0. Output written to `/logs/verifier/reward.json`.

## Workflow

1. **Read** `/app/instruction.md` for the full upstream spec.
2. **Plan** — `submit_plan` with one subtask covering the implementation
   (correctness first, then optimize for benchmarks).
3. **Implement** — write `.s` / `.asm` files under `/app/asm-port/`,
   assemble + link with `nasm`/`as`/`ld` to produce `libexpat.so`.
4. **Submit** — `submit_subtask` runs the full verifier and returns a
   normalized reward.

**Remember:** correctness gates performance. A `.so` that exports the right
symbols but fails most parser tests scores ~0. Aim for the parser core
working end-to-end, then optimize.
""".strip()


def _load_upstream_instruction() -> str:
    upstream = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "libexpat-to-x86asm"
        / "instruction.md"
    )
    if upstream.is_file():
        return upstream.read_text()
    return LIBEXPAT_TRAINING_INSTRUCTION


def _common_kwargs() -> dict:
    return {
        "task_name": "libexpat-to-x86asm",
        "docker_image": "frontier-swe-libexpat-to-x86asm:latest",
        "workspace_dir": "/app/asm-port",
        # No agent-side build step; the verifier handles all compilation.
        "build_command": "true",
        "gate_script_path": "/app/gate_checks.sh",
        "visible_test_command": "bash /opt/verifier/test.sh",
        "visible_test_total": 1,
        "l1_score_mode": "reward_json_score",
        "reward_json_path": "/logs/verifier/reward.json",
        "reward_json_score_field": "score",
        # Oracle (full correctness + ~parity perf) ≈ 1.0; stub fail = 0.0.
        # Direct identity mapping. Agents that beat reference C will clamp
        # at 1.0 — re-tune after observing real runs if that becomes common.
        "reward_json_score_anchors": (0.0, 1.0),
        "reward_json_score_higher_is_better": True,
        "gate_threshold": 0.6,
        "max_subtasks": 1,
        "task_domain": "systems programming / x86-64 assembly / XML parsing",
        "cpus": 4,
        "memory_mb": 8192,
    }


def libexpat_training_config() -> TaskConfig:
    return TaskConfig(
        instruction=LIBEXPAT_TRAINING_INSTRUCTION,
        max_attempts_per_subtask=3,
        episode_timeout_s=3600.0,
        per_turn_timeout_s=600.0,
        l1_timeout_s=1500.0,
        task_description=(
            "Reimplement libexpat 2.6.4 in x86-64 assembly. Scored on "
            "correctness (50%, expat test suite pass rate) and performance "
            "(50%, parsing speed vs reference C build)."
        ),
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh. Reads /logs/verifier/reward.json. "
            "Hard-fails (no .so / anti-cheat / gcc unavailable / zero correctness) "
            "set score=0.0. Otherwise score is 0.5*correctness + 0.5*performance, "
            "normalized via score / 1.0 clamped to [0, 1]. Subscores include "
            "correctness and performance per benchmark doc."
        ),
        **_common_kwargs(),
    )


def libexpat_demo_config() -> TaskConfig:
    return TaskConfig(
        instruction=_load_upstream_instruction(),
        max_attempts_per_subtask=5,
        episode_timeout_s=7200.0,
        per_turn_timeout_s=900.0,
        l1_timeout_s=2400.0,
        task_description=(
            "Reimplement libexpat in x86-64 assembly (demo mode: longer "
            "budgets and more attempts)."
        ),
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh. Reads /logs/verifier/reward.json. "
            "Hard-fails set score=0.0. Otherwise 0.5*correctness + 0.5*performance."
        ),
        **_common_kwargs(),
    )
