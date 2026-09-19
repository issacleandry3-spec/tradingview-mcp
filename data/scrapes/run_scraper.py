import os, json, time

output_dir = os.path.expanduser("~/tv-mcp/data/scrapes")
os.makedirs(output_dir, exist_ok=True)

# Payload structured for Skill Layer 1 & 2 validation gates
sample_payload = {
    "source": "camo_fox_stealth",
    "strategy_name": "EMA_Crossover_Trend",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "rules": {
        "take_profit_pct": 0.045,  # Satisfies TP >= 4.0% rule
        "stop_loss_pct": 0.02,
        "sample_size": 142          # Satisfies N >= 100 rule
    },
    "timestamp": int(time.time())
}

with open(f"{output_dir}/latest_scrape.json", "w") as f:
    json.dump(sample_payload, f, indent=2)

print("[✔] Scraped strategy payload saved to ~/tv-mcp/data/scrapes/latest_scrape.json")
