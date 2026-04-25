"""Quick test of all profiles."""
from rtmdk.config import RTMDKConfig
profiles = ['local', 'production', 'research', 'enterprise', 'agent', 'legal', 'medical', 'streaming']
for p in profiles:
    cfg = getattr(RTMDKConfig, p)()
    print(f"{p}: engrams={cfg.enable_engrams}, dream={cfg.offline_dreaming}, causal={cfg.causal_traversal}, ssm={cfg.ssm_dynamics}")
print("All profiles OK")
