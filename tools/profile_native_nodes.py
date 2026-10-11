"""NVTX request markers around the existing native driver, for diagnostics only.

Run this script under nsys with --trace=cuda,nvtx --cuda-graph-trace=node.
The llama.cpp binary, request payload, and normal native driver remain unchanged.
"""
import json
import sys
import time
from datetime import datetime, timezone

import nvtx
import native_benchmark as benchmark


def main():
    args = benchmark._parse_args()
    original = benchmark._stream_completion
    records = []

    def marked_request(*positional, **keywords):
        index = len(records)
        kind = "warmup" if index < args.warmup else "formal"
        label = f"{kind}_{index + 1 if kind == 'warmup' else index - args.warmup + 1}"
        record = {"label": label, "started_utc": datetime.now(timezone.utc).isoformat()}
        records.append(record)
        with nvtx.annotate(label, domain="heterollm_diagnostic"):
            result, start, end = original(*positional, **keywords)
            record.update(client_start_ns=start, client_end_ns=end,
                          first_visible_ns=result["first_visible_ns"],
                          last_visible_ns=result["last_visible_ns"])
        # Permit CUPTI periodic activity flushing before the server is terminated.
        time.sleep(1.0)
        return result, start, end

    benchmark._stream_completion = marked_request
    result = benchmark.run(args)
    result["schema"] = "heterollm.native-node-diagnostic/v1"
    result["diagnostic_only"] = True
    result["normal_native_reference"] = False
    result["nvtx_requests"] = records
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
