# Unified Quantitative & Execution Skill Architecture

## Meta-Rule: Automatic Skill Integration & Cross-Verification Protocol
Whenever a new skill, tool, script, or system directive is added to this environment:
1. **Pipeline Hook**: Claude MUST map the new skill into one of the 5 execution layers below before executing trades or running backtests.
2. **Cross-Layer Validation**: Every output must be validated by adjacent skills (e.g., an entry signal must pass the Fee Floor and Sample Size gates).
3. **Fail-Safe Cascade**: If any skill in the pipeline fails its gate, execution automatically cascades back to parameter relaxation or safety aborts.

---

## The 5-Layer Skill Interlock Matrix

### 1. Pre-Flight Fee & Execution Boundary Check
- Verify expected gross yield per trade > 2x (round-trip slippage + priority gas fees).
- If gross capture < $5.00 on DEX or < 0.3% on CEX, abort micro-scalping and re-route execution to liquid CEX pairs (BTC/ETH/SOL) or enforce minimum TP thresholds (TP >= 4.0%).

### 2. Statistical Sample Size & Significance Gate
- Reject any strategy refinement, signal, or baseline update where N < 100 total trades across multi-year backtests (minimum ~12 trades/year).
- Sample sizes where N < 30 per year are statistically invalid due to overfitting and sample bias.

### 3. Risk-Adjusted Evaluation & Auto-Loop
- Prioritize Sharpe Ratio and Max Drawdown over raw CAGR.
- If raw CAGR gates fail, automatically relax the CAGR hurdle (ΔCAGR >= -2pp) to evaluate strategies that cut max drawdown significantly (e.g., cutting drawdown from -64% to -32%).
- Record all grid-search runs in `scalper_journal.csv` and update `baseline.json` ONLY when both Skill 2 (N >= 100) and Skill 3 (Sharpe improvement) pass.

### 4. Hybrid Position Management
- Divide position execution into standard mode vs. emergency mode:
  - Standard Mode: 3-tranche profit laddering (+25%, +50%, +100%) to lock in gains.
  - Emergency Mode: Instant 100% position dump.

### 5. Mempool Pre-Execution & Anti-Rug MEV Skill
- Integrate async WebSocket/gRPC streams (Helius/Triton for Solana, Flashbots/MEV-Share for EVM) to parse incoming block instructions before landing.
- On detection of `remove_liquidity`, `burn_lp`, `set_freeze_authority`, or dev wallet dumps > 5% supply:
  - Instantly override and cancel all active Skill 4 ladder orders.
  - Fire a pre-signed 100% sell bundle directly to block validators (Jito on Solana, Flashbots on EVM) with dynamic priority tip fees to land ahead of or alongside the malicious transaction slot.
  - Keys MUST be loaded via `.env` environment variables (`ETH_PRIVATE_KEY`) rather than requested as raw prompts.

---

## Dynamic Skill Registry (Auto-Updated)
- [x] Fee & Execution Boundary (Active)
- [x] Statistical Sample Size Gate (Active)
- [x] Risk-Adjusted Sharpe/DD Optimizer (Active)
- [x] Hybrid Profit Laddering (Active)
- [x] Anti-Rug Solana / Jito MEV Execution (Active) → `antirug_listener.py`
- [x] Anti-Rug EVM / Flashbots + MEV-Share Execution (Active) → `antirug_evm.py`
- [x] Rigorous Step-by-Step Math Solver & Verifier (Active) → `skills/math_solver.md`
- [ ] *[Future Skills Auto-Register Here]*
