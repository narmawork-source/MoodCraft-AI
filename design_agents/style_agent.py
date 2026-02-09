from typing import Dict, List


def analyze_style_dna(text_prompt: str) -> Dict:
    prompt = (text_prompt or '').lower()
    tags = [t for t in ['japandi','modern','coastal','mid-century','minimalist'] if t in prompt]
    if not tags:
        tags = ['modern','minimalist']
    return {
        'style_tags': tags,
        'palette': [
            {'hex':'#f5f1ea','usage':'base'},
            {'hex':'#d8c9b1','usage':'secondary'},
            {'hex':'#8b6f47','usage':'accent'},
        ],
        'materials': ['oak','linen','rattan'],
        'decor_types': ['rug','lamp','console','chair'],
        'do_constraints': ['keep warm neutrals','use natural textures'],
        'dont_constraints': ['avoid neon','avoid clutter'],
    }


def embed_stub(text: str) -> List[float]:
    vals = [float((ord(c) % 11)) for c in text[:32]]
    if not vals:
        return [0.0]
    return vals
