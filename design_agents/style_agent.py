from typing import Dict, List


STYLE_PRESETS = {
    "modern": {
        "style_tags": ["modern", "clean-lines"],
        "palette": [{"hex": "#f2f2f0", "usage": "base"}, {"hex": "#c9ced3", "usage": "secondary"}, {"hex": "#2f353d", "usage": "accent"}],
        "materials": ["oak", "glass", "brushed metal"],
        "decor_types": ["sofa", "coffee table", "rug", "lamp"],
    },
    "industrial": {
        "style_tags": ["industrial", "urban-loft"],
        "palette": [{"hex": "#2e2e2e", "usage": "base"}, {"hex": "#8c7a6b", "usage": "secondary"}, {"hex": "#b35a3c", "usage": "accent"}],
        "materials": ["black steel", "concrete", "reclaimed wood"],
        "decor_types": ["floor lamp", "metal shelf", "leather chair", "rug"],
    },
    "contemporary": {
        "style_tags": ["contemporary", "soft-geometry"],
        "palette": [{"hex": "#f6f4ef", "usage": "base"}, {"hex": "#d5d0c7", "usage": "secondary"}, {"hex": "#5f6b73", "usage": "accent"}],
        "materials": ["linen", "walnut", "matte metal"],
        "decor_types": ["sofa", "console", "wall art", "lamp"],
    },
    "traditional": {
        "style_tags": ["traditional", "classic"],
        "palette": [{"hex": "#f4eadb", "usage": "base"}, {"hex": "#8f6f4f", "usage": "secondary"}, {"hex": "#4b2f23", "usage": "accent"}],
        "materials": ["mahogany", "cotton", "brass"],
        "decor_types": ["armchair", "side table", "curtains", "rug"],
    },
    "minimalist": {
        "style_tags": ["minimalist", "decluttered"],
        "palette": [{"hex": "#ffffff", "usage": "base"}, {"hex": "#dfdfdf", "usage": "secondary"}, {"hex": "#222222", "usage": "accent"}],
        "materials": ["light wood", "cotton", "matte black steel"],
        "decor_types": ["sofa", "low table", "single lamp", "storage"],
    },
    "scandinavian": {
        "style_tags": ["scandinavian", "airy"],
        "palette": [{"hex": "#faf8f2", "usage": "base"}, {"hex": "#cfd8d1", "usage": "secondary"}, {"hex": "#7b8c7a", "usage": "accent"}],
        "materials": ["birch", "wool", "linen"],
        "decor_types": ["rug", "light wood chair", "floor lamp", "plants"],
    },
    "japandi": {
        "style_tags": ["japandi", "zen-minimal"],
        "palette": [{"hex": "#f1ede4", "usage": "base"}, {"hex": "#b7aa95", "usage": "secondary"}, {"hex": "#5f4c3d", "usage": "accent"}],
        "materials": ["oak", "linen", "rattan"],
        "decor_types": ["low table", "paper lamp", "neutral rug", "storage"],
    },
    "bohemian": {
        "style_tags": ["bohemian", "eclectic"],
        "palette": [{"hex": "#f7f0e2", "usage": "base"}, {"hex": "#c98b5d", "usage": "secondary"}, {"hex": "#6d4b3a", "usage": "accent"}],
        "materials": ["rattan", "jute", "textiles"],
        "decor_types": ["pattern rug", "floor cushions", "plants", "wall art"],
    },
    "mid-century": {
        "style_tags": ["mid-century", "retro-modern"],
        "palette": [{"hex": "#f2e8d8", "usage": "base"}, {"hex": "#a4724a", "usage": "secondary"}, {"hex": "#1f4d5a", "usage": "accent"}],
        "materials": ["walnut", "leather", "brass"],
        "decor_types": ["console", "accent chair", "table lamp", "rug"],
    },
    "coastal": {
        "style_tags": ["coastal", "breezy"],
        "palette": [{"hex": "#f6fbff", "usage": "base"}, {"hex": "#b7d8e8", "usage": "secondary"}, {"hex": "#6e8b94", "usage": "accent"}],
        "materials": ["light oak", "cotton", "seagrass"],
        "decor_types": ["woven rug", "light curtains", "coastal art", "lamp"],
    },
    "farmhouse": {
        "style_tags": ["farmhouse", "rustic-cozy"],
        "palette": [{"hex": "#f7f3ea", "usage": "base"}, {"hex": "#b8a48c", "usage": "secondary"}, {"hex": "#5e4b3c", "usage": "accent"}],
        "materials": ["reclaimed wood", "cotton", "wrought iron"],
        "decor_types": ["wood table", "lantern lamp", "woven rug", "storage"],
    },
    "luxury": {
        "style_tags": ["luxury", "premium"],
        "palette": [{"hex": "#f8f6f2", "usage": "base"}, {"hex": "#c9b18b", "usage": "secondary"}, {"hex": "#1e1f24", "usage": "accent"}],
        "materials": ["marble", "velvet", "brass"],
        "decor_types": ["statement sofa", "designer lamp", "art piece", "mirror"],
    },
}


def _match_style_key(prompt: str) -> str:
    aliases = {
        "midcentury": "mid-century",
        "mid century": "mid-century",
    }
    norm = prompt.lower()
    for alias, key in aliases.items():
        if alias in norm:
            return key
    for key in STYLE_PRESETS:
        if key in norm:
            return key
    return "modern"


def analyze_style_dna(text_prompt: str) -> Dict:
    prompt = (text_prompt or "").lower()
    style_key = _match_style_key(prompt)
    base = STYLE_PRESETS[style_key]
    return {
        "style_tags": base["style_tags"],
        "palette": base["palette"],
        "materials": base["materials"],
        "decor_types": base["decor_types"],
        "do_constraints": [f"follow {style_key} cues", "keep palette and materials cohesive"],
        "dont_constraints": ["avoid style mixing", "avoid visual clutter"],
    }


def embed_stub(text: str) -> List[float]:
    vals = [float((ord(c) % 11)) for c in text[:32]]
    if not vals:
        return [0.0]
    return vals
