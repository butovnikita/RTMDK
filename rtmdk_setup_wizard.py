"""
rtmdk_setup_wizard.py — Interactive Setup Wizard for RTMDK.

Guides users through configuration:
1. Select provider (LM Studio, OpenRouter, OpenAI, Custom)
2. Select models
3. Configure embedder
4. Select preset
5. Save configuration

Usage:
    python rtmdk_setup_wizard.py [--docker]
"""

import os
import sys
import json
from pathlib import Path

# Available presets
PRESETS = {
    "local": {
        "name": "Local (LM Studio)",
        "description": "Single user, LM Studio on localhost, built-in embedder",
        "config": {
            "RTMDK_API_PROVIDER": "lm_studio",
            "LM_STUDIO_URL": "http://localhost:12345/v1",
            "RTMDK_EMBED_MODEL": "nomic-embed-text-v1.5",
            "RTMDK_AUTO_SAVE": 60
        }
    },
    "production": {
        "name": "Production Server",
        "description": "Multi-user, external API, Redis/PostgreSQL ready",
        "config": {
            "RTMDK_API_PROVIDER": "openrouter",
            "RTMDK_AUTO_SAVE": 30,
            "RTMDK_MEMORY_FILE": "./data/memory.json"
        }
    },
    "sillytavern": {
        "name": "SillyTavern Integration",
        "description": "Pre-configured for SillyTavern with memory proxy",
        "config": {
            "RTMDK_API_PROVIDER": "lm_studio",
            "LM_STUDIO_URL": "http://localhost:12345/v1",
            "RTMDK_EMBED_MODEL": "nomic-embed-text-v1.5"
        }
    },
    "agent": {
        "name": "IDE Agent Mode",
        "description": "Optimized for code generation with memory",
        "config": {
            "RTMDK_API_PROVIDER": "lm_studio",
            "LM_STUDIO_URL": "http://localhost:12345/v1",
            "RTMDK_EMBED_MODEL": "nomic-embed-text-v1.5",
            "RTMDK_AUTO_SAVE": 30
        }
    }
}

def print_banner():
    print("\n" + "=" * 60)
    print("  >>> RTMDK Setup Wizard")
    print("=" * 60)
    print("  This wizard will help you configure RTMDK for your use case.")
    print("=" * 60 + "\n")

def select_preset():
    """Let user select a preset."""
    print("\n[PKG] Available Presets:")
    print("-" * 40)
    
    presets = list(PRESETS.items())
    for i, (key, preset) in enumerate(presets, 1):
        print(f"  {i}. {preset['name']}")
        print(f"     {preset['description']}")
    
    while True:
        try:
            choice = input(f"\nSelect preset (1-{len(presets)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                return presets[idx]
            print(f"  Please enter a number between 1 and {len(presets)}")
        except ValueError:
            print("  Please enter a valid number")

def configure_provider(preset_key, preset_data):
    """Configure provider-specific settings."""
    config = preset_data['config'].copy()
    
    print(f"\n[CFG] Configuring: {preset_data['name']}")
    print("-" * 40)
    
    if preset_key == 'local' or preset_key == 'sillytavern' or preset_key == 'agent':
        # LM Studio configuration
        lm_url = input(f"  LM Studio URL [{config.get('LM_STUDIO_URL', 'http://localhost:12345/v1')}]: ").strip()
        if lm_url:
            config['LM_STUDIO_URL'] = lm_url
        
        # Embedder selection
        print("\n  Available Embedders:")
        print("    1. nomic-embed-text-v1.5 (default, 768d)")
        print("    2. all-MiniLM-L6-v2 (384d)")
        print("    3. text-embedding-3-small (1536d)")
        
        while True:
            emb_choice = input("\n  Select embedder (1-3): ").strip()
            embedders = {
                '1': 'nomic-embed-text-v1.5',
                '2': 'all-MiniLM-L6-v2',
                '3': 'text-embedding-3-small'
            }
            if emb_choice in embedders:
                config['RTMDK_EMBED_MODEL'] = embedders[emb_choice]
                break
            print("  Please enter 1, 2, or 3")
    
    elif preset_key == 'production':
        # External API configuration
        print("\n  Select API Provider:")
        print("    1. OpenRouter")
        print("    2. OpenAI")
        print("    3. Anthropic")
        print("    4. Custom URL")
        
        providers = {
            '1': 'openrouter',
            '2': 'openai',
            '3': 'anthropic',
            '4': 'custom'
        }
        
        while True:
            prov_choice = input("\n  Select provider (1-4): ").strip()
            if prov_choice in providers:
                config['RTMDK_API_PROVIDER'] = providers[prov_choice]
                break
            print("  Please enter 1, 2, 3, or 4")
        
        # API key
        api_key = input("\n  API Key: ").strip()
        if api_key:
            if config['RTMDK_API_PROVIDER'] == 'openrouter':
                config['OPENROUTER_API_KEY'] = api_key
            elif config['RTMDK_API_PROVIDER'] == 'openai':
                config['OPENAI_API_KEY'] = api_key
            elif config['RTMDK_API_PROVIDER'] == 'anthropic':
                config['ANTHROPIC_API_KEY'] = api_key
    
    return config

def save_config(config, is_docker=False):
    """Save configuration to file."""
    if is_docker:
        # Save as .env for Docker
        env_path = ".env"
        with open(env_path, 'w') as f:
            f.write("# RTMDK Configuration - Generated by Setup Wizard\n")
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        print(f"\n[OK] Configuration saved to {env_path}")
        print("   Start with: docker-compose up -d")
    else:
        # Save as .env for local
        env_path = ".env"
        with open(env_path, 'w') as f:
            f.write("# RTMDK Configuration - Generated by Setup Wizard\n")
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        print(f"\n[OK] Configuration saved to {env_path}")
        print("   Start with: python rtmdk_server.py")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTMDK Setup Wizard")
    parser.add_argument("--docker", action="store_true", help="Configure for Docker")
    args = parser.parse_args()
    
    print_banner()
    
    # Step 1: Select preset
    preset_key, preset_data = select_preset()
    
    # Step 2: Configure provider
    config = configure_provider(preset_key, preset_data)
    
    # Step 3: Review configuration
    print("\n[SUMMARY] Configuration Summary:")
    print("-" * 40)
    for key, value in config.items():
        # Mask API keys
        if 'API_KEY' in key and value:
            value = value[:8] + "..." if len(value) > 8 else "***"
        print(f"  {key}: {value}")

    # Step 4: Save
    confirm = input("\n[SAVE] Save configuration? (y/n): ").strip().lower()
    if confirm == 'y':
        save_config(config, is_docker=args.docker)

        print("\n" + "=" * 60)
        print("  [DONE] Setup Complete!")
        print("=" * 60)
        
        if args.docker:
            print("\n  Next steps:")
            print("    1. docker-compose up -d")
            print("    2. Open http://localhost:8080/dashboard")
            print("    3. Configure SillyTavern: http://localhost:5000/v1")
        else:
            print("\n  Next steps:")
            print("    1. python rtmdk_server.py")
            print("    2. Open http://localhost:8080/dashboard")
            print("    3. For SillyTavern: python rtmdk_st_proxy.py")
        
        print("=" * 60 + "\n")
    else:
        print("\n[FAIL] Configuration cancelled.")

if __name__ == "__main__":
    main()
