from typing import List, Dict, Any, Optional

def make_attr(key: str, val: Any) -> Dict[str, Any]:
    if isinstance(val, int):
        return {"key": key, "value": {"intValue": val}}
    elif isinstance(val, bool):
        return {"key": key, "value": {"boolValue": val}}
    else:
        return {"key": key, "value": {"stringValue": str(val)}}

def build_otlp_trace(spans_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": spans_list
                    }
                ]
            }
        ]
    }
