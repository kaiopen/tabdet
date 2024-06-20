from typing import Any, Dict, Sequence


def process_logs(logs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def add(src: Dict[str, Any], dst: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, Dict):
                add(v, dst[k])
            else:
                dst[k] += v

    def div(d: Dict[str, Any], div: float) -> None:
        for k, v in d.items():
            if isinstance(v, Dict):
                div(v, div)
            else:
                d[k] = v / div

    log = logs[0]
    for _log in logs[1:]:
        add(_log, log)

    div(log, len(logs))
    return log
