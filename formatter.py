from typing import Dict, Any
from utils import logger


def format_output(misinfo: Dict[str, Any], asset_assessment: Dict[str, Any], sources: list[str] | None = None) -> Dict[str, Any]:
    try:
        bucket = asset_assessment.get('risk', {}).get('bucket', 'green')
        score = asset_assessment.get('risk', {}).get('score', 0)
        summary_parts = []
        if misinfo:
            summary_parts.append(f"Misinformation: {misinfo.get('label')} ({misinfo.get('score'):.2f})")
        reasons = asset_assessment.get('risk', {}).get('reasons', [])
        if reasons:
            summary_parts.append(', '.join(reasons))
        summary = '. '.join(summary_parts)[:500]
        out = {
            'risk_bucket': bucket,
            'risk_score': score,
            'summary': summary,
            'sources': sources or [],
            'details': {
                'misinfo': misinfo,
                'asset': asset_assessment,
            }
        }
        return out
    except Exception:
        logger.exception('Failed to format output')
        return {'risk_bucket': 'green', 'risk_score': 0, 'summary': '', 'sources': [], 'details': {}}
