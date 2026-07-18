import json

from audiosr.utils import default_audioldm_config

cfg = default_audioldm_config("basic")
p = cfg["model"]["params"]
print("first_stage_key:", p.get("first_stage_key"))
print("cond_stage_config:", json.dumps(p.get("cond_stage_config"), indent=1, default=str))
print("unet_config params:", json.dumps(p.get("unet_config", {}).get("params"), indent=1, default=str))
print("first_stage_config params keys:", json.dumps(p.get("first_stage_config", {}).get("params", {}), indent=1, default=str)[:800])
print("top-level:", {k: v for k, v in p.items() if not isinstance(v, dict)})
