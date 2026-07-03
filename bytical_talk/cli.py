"""bytical-talk command-line interface.

    bytical-talk env-check
    bytical-talk direct     --script "..."            # LLM performance plan (JSON)
    bytical-talk autoconfig --video in.mp4            # analyze video -> settings
    bytical-talk qc         --video out.mp4           # quality review
    bytical-talk generate   --script "..." --checkpoint c.pth --dataset d/ \
                            --audio a.wav --out out.mp4 [--reference ref.mp4]
"""

from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_env_check(_args) -> int:
    import os

    from .brain.llm import LLMClient

    llm = LLMClient()
    info = {
        "llm_provider": llm.config.provider,
        "llm_available": llm.available,
        "chat_model": llm._chat_model,
        "embed_model": llm._embed_model,
        "upstream_present": os.path.isdir("upstream/synctalk2d"),
    }
    _print(info)
    return 0 if llm.available else 1


def cmd_direct(args) -> int:
    from .brain.director import Director

    plan = Director().direct(args.script)
    _print(plan.to_dict())
    return 0


def cmd_autoconfig(args) -> int:
    from .brain.autoconfig import auto_config

    stats, cfg = auto_config(args.video)
    _print({"stats": stats.__dict__, "config": cfg.to_dict()})
    return 0


def cmd_qc(args) -> int:
    from .brain.autoconfig import RenderConfig
    from .brain.qc import SelfQC, measure

    metrics = measure(args.video)
    report = SelfQC().review(metrics, RenderConfig())
    _print({"metrics": metrics.__dict__, "passed": report.passed,
            "issues": report.issues, "suggestion": report.suggestion,
            "rationale": report.rationale})
    return 0 if report.passed else 2


def cmd_generate(args) -> int:
    from .pipeline import Pipeline

    result = Pipeline(max_retries=args.max_retries).generate(
        script=args.script, checkpoint=args.checkpoint, dataset_dir=args.dataset,
        out_path=args.out, reference_video=args.reference, audio_path=args.audio,
        asr=args.asr, upstream_dir=args.upstream,
    )
    _print({"output": result.output_path, "attempts": result.attempts,
            "qc_passed": result.qc_passed, "config": result.config.to_dict(),
            "plan": result.plan.to_dict()})
    return 0 if result.qc_passed else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bytical-talk")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env-check").set_defaults(fn=cmd_env_check)

    d = sub.add_parser("direct")
    d.add_argument("--script", required=True)
    d.set_defaults(fn=cmd_direct)

    a = sub.add_parser("autoconfig")
    a.add_argument("--video", required=True)
    a.set_defaults(fn=cmd_autoconfig)

    q = sub.add_parser("qc")
    q.add_argument("--video", required=True)
    q.set_defaults(fn=cmd_qc)

    g = sub.add_parser("generate")
    g.add_argument("--script", required=True)
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--dataset", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--audio", default=None)
    g.add_argument("--reference", default=None)
    g.add_argument("--asr", default="ave", choices=["ave", "hubert"])
    g.add_argument("--upstream", default="upstream/synctalk2d")
    g.add_argument("--max-retries", dest="max_retries", type=int, default=1)
    g.set_defaults(fn=cmd_generate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
