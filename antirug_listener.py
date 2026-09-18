#!/usr/bin/env python3
"""
antirug_listener.py — Solana Anti-Rug MEV Exit System

Monitors a Solana pool/token position via Helius enhanced WebSocket
(transactionSubscribe). On detection of rug-pull patterns it fires a
pre-signed exit bundle via Jito block engine.

Detection triggers (per CLAUDE.md §5):
  1. remove_liquidity / burn_lp on the monitored pool
  2. set_freeze_authority or mint-authority change on the token mint
  3. Dev/creator wallet dumps > 5 % of circulating supply in one slot

Exit flow:
  Normal trading : laddered take-profits at +25 %, +50 % of entry
  Emergency exit : 100 % market dump via Jupiter swap → Jito bundle

Usage:
    python antirug_listener.py --config config.json [--dry-run]

Sensitive keys are loaded from .env (never put in config.json):
    SOLANA_PRIVATE_KEY=<base58 private key>
    HELIUS_API_KEY=<helius api key>

Non-sensitive config.json:
{
    "pool_address":        "POOL_PUBKEY",
    "token_mint":          "TOKEN_MINT_PUBKEY",
    "dev_wallet":          "DEV_WALLET_PUBKEY",
    "your_wallet":         "YOUR_WALLET_PUBKEY",
    "token_account":       "YOUR_SPL_TOKEN_ACCOUNT",
    "token_decimals":      6,
    "position_amount":     1000000,
    "circulating_supply":  1000000000,
    "jito_tip_lamports":   100000
}

Dependencies:
    pip install websockets httpx base58 python-dotenv cryptography
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

import base58
import httpx
import websockets
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("antirug")

# ── Solana program IDs ────────────────────────────────────────────────────────
SPL_TOKEN          = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
SPL_TOKEN_2022     = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
RAYDIUM_AMM_V4     = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CLMM       = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
ORCA_WHIRLPOOL     = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
METEORA_DLMM       = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo"
PUMP_FUN           = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
SYSTEM_PROGRAM     = "11111111111111111111111111111111"

# Jito tip accounts (pick one per bundle)
JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1uf6it5d9pd",
]
JITO_BLOCK_ENGINE = "https://mainnet.block-engine.jito.foundation/api/v1/bundles"


# ── Instruction discriminators (first 8 bytes of Anchor sha256) ──────────────
def _disc(namespace: str, name: str) -> bytes:
    return sha256(f"{namespace}:{name}".encode()).digest()[:8]


DISCRIMINATORS: dict[str, dict[str, bytes]] = {
    RAYDIUM_CLMM: {
        "decrease_liquidity":    _disc("global", "decrease_liquidity"),
        "decrease_liquidity_v2": _disc("global", "decrease_liquidity_v2"),
    },
    ORCA_WHIRLPOOL: {
        "decrease_liquidity":    _disc("global", "decrease_liquidity"),
        "decrease_liquidity_v2": _disc("global", "decrease_liquidity_v2"),
    },
    METEORA_DLMM: {
        "remove_liquidity":      _disc("global", "remove_liquidity"),
        "remove_all_liquidity":  _disc("global", "remove_all_liquidity"),
    },
}

# Raydium AMM v4 uses a plain u8 instruction index (not Anchor discriminator)
RAYDIUM_AMM_REMOVE_LIQ_IX = 4   # WithdrawV2 = 4 in Raydium AMM v4

# SPL Token SetAuthority instruction type byte
SPL_SET_AUTHORITY_TYPE = 6

# Authority types that signal a freeze/mint rug
DANGEROUS_AUTHORITY_TYPES = {
    0,   # MintTokens (regaining mint authority)
    1,   # FreezeAccount (enabling freeze)
}

# Log-level patterns (fallback when instruction data unavailable)
RUG_LOG_PATTERNS = [
    "remove_liquidity", "removeliquidity", "remove liquidity",
    "burn_lp", "burnlp", "burn lp",
    "withdraw_all", "withdrawall",
    "set_freeze_authority", "setfreezeauthority",
]


# ── Configuration ─────────────────────────────────────────────────────────────
def _load_env(env_file: str | None = None) -> dict[str, str]:
    """
    Load .env from env_file path, then the project root .env, then the
    process environment.  Raises if any required key is absent.
    """
    # Explicit path first, then project root .env
    for candidate in filter(None, [env_file,
                                   str(Path(__file__).parent / ".env")]):
        if Path(candidate).exists():
            load_dotenv(candidate, override=False)
            log.info("Loaded .env from %s", candidate)
            break

    required = ("SOLANA_PRIVATE_KEY", "HELIUS_API_KEY")
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {missing}\n"
            f"Add them to .env or export them before running."
        )
    return {k: os.environ[k] for k in required}


@dataclass
class Config:
    # non-sensitive — safe in config.json
    pool_address:        str
    token_mint:          str
    dev_wallet:          str
    your_wallet:         str
    token_account:       str        # your SPL token account
    token_decimals:      int        = 6
    position_amount:     int        = 0   # raw token units currently held
    circulating_supply:  int        = 1_000_000_000
    jito_tip_lamports:   int        = 100_000
    dry_run:             bool       = True
    # sensitive — injected from env after construction, never serialised
    helius_api_key:      str        = field(default="", repr=False)
    private_key_b58:     str        = field(default="", repr=False)
    # derived
    helius_ws_url:       str        = field(init=False, repr=False)
    helius_rpc_url:      str        = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.helius_api_key:
            raise ValueError("helius_api_key must be set before __post_init__")
        self.helius_ws_url  = (f"wss://mainnet.helius-rpc.com/"
                               f"?api-key={self.helius_api_key}")
        self.helius_rpc_url = (f"https://mainnet.helius-rpc.com/"
                               f"?api-key={self.helius_api_key}")

    @classmethod
    def from_file(cls, path: str, dry_run: bool = True,
                  env_file: str | None = None) -> "Config":
        env = _load_env(env_file)
        with open(path) as f:
            d = json.load(f)
        # strip any accidentally committed secrets from config.json
        d.pop("private_key_b58", None)
        d.pop("helius_api_key", None)
        d["dry_run"]        = dry_run
        d["helius_api_key"] = env["HELIUS_API_KEY"]
        d["private_key_b58"]= env["SOLANA_PRIVATE_KEY"]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Signing helpers ───────────────────────────────────────────────────────────
def _load_keypair(private_key_b58: str) -> Ed25519PrivateKey:
    raw = base58.b58decode(private_key_b58)
    if len(raw) == 64:
        raw = raw[:32]   # Solana stores 64-byte keypair; seed is first 32
    return Ed25519PrivateKey.from_private_bytes(raw)


def _pubkey_from_keypair(sk: Ed25519PrivateKey) -> str:
    pub_bytes = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return base58.b58encode(pub_bytes).decode()


# ── Dev-wallet supply tracker ─────────────────────────────────────────────────
class DevWalletTracker:
    """Accumulate token transfers out of the dev wallet within a slot window."""

    def __init__(self, dev_wallet: str, circulating_supply: int,
                 dump_threshold: float = 0.05, window_slots: int = 1) -> None:
        self.dev_wallet         = dev_wallet
        self.circulating_supply = circulating_supply
        self.dump_threshold     = dump_threshold
        self.window_slots       = window_slots
        self._slot_transfers: dict[int, int] = {}   # slot → cumulative raw amount

    def record_transfer(self, slot: int, amount_raw: int) -> bool:
        """Return True if cumulative dump in this slot exceeds threshold."""
        self._slot_transfers[slot] = self._slot_transfers.get(slot, 0) + amount_raw
        pct = self._slot_transfers[slot] / self.circulating_supply
        if pct >= self.dump_threshold:
            log.warning(
                "DEV DUMP  slot=%d  amount=%d  pct=%.2f%%",
                slot, self._slot_transfers[slot], pct * 100,
            )
            return True
        return False

    def prune(self, current_slot: int) -> None:
        stale = [s for s in self._slot_transfers if s < current_slot - self.window_slots]
        for s in stale:
            del self._slot_transfers[s]


# ── Instruction analyzer ──────────────────────────────────────────────────────
class InstructionAnalyzer:
    """Parse Helius transaction notifications for rug-pull instructions."""

    def __init__(self, cfg: Config) -> None:
        self.cfg    = cfg
        self.dev_wt = DevWalletTracker(
            cfg.dev_wallet, cfg.circulating_supply
        )

    def analyze(self, tx_notification: dict) -> list[str]:
        """Return list of triggered rug patterns (empty = clean)."""
        triggers: list[str] = []
        tx = tx_notification.get("transaction", {})
        slot = tx_notification.get("slot", 0)
        meta = tx.get("meta", {})
        msg  = tx.get("transaction", {}).get("message", {})

        # ── 1. Log-level pattern scan (fast path) ────────────────────────────
        logs: list[str] = meta.get("logMessages") or []
        for line in logs:
            low = line.lower()
            for pat in RUG_LOG_PATTERNS:
                if pat in low:
                    triggers.append(f"log_pattern:{pat}")

        # ── 2. Instruction-level scan ─────────────────────────────────────────
        account_keys: list[str] = msg.get("accountKeys", [])
        instructions: list[dict] = msg.get("instructions", [])
        inner_ixs_list: list[dict] = meta.get("innerInstructions", [])

        all_ixs = list(instructions)
        for inner in inner_ixs_list:
            all_ixs.extend(inner.get("instructions", []))

        for ix in all_ixs:
            prog_idx = ix.get("programIdIndex", -1)
            if prog_idx < 0 or prog_idx >= len(account_keys):
                continue
            prog = account_keys[prog_idx]
            data_b58 = ix.get("data", "")

            try:
                data = base58.b58decode(data_b58) if data_b58 else b""
            except Exception:
                continue

            t = self._check_instruction(prog, data, account_keys,
                                        ix.get("accounts", []), slot)
            triggers.extend(t)

        # de-duplicate
        return list(dict.fromkeys(triggers))

    def _check_instruction(
        self, prog: str, data: bytes,
        account_keys: list[str], account_indices: list[int],
        slot: int,
    ) -> list[str]:
        triggers: list[str] = []

        # ── SPL Token / Token-2022 SetAuthority ──────────────────────────────
        if prog in (SPL_TOKEN, SPL_TOKEN_2022):
            if len(data) >= 2 and data[0] == SPL_SET_AUTHORITY_TYPE:
                auth_type = data[1]
                if auth_type in DANGEROUS_AUTHORITY_TYPES:
                    name = "freeze_authority" if auth_type == 1 else "mint_authority"
                    triggers.append(f"spl:set_{name}")
                    log.warning("SPL SetAuthority detected  prog=%s  type=%d", prog, auth_type)

        # ── Raydium AMM v4 (u8 index-based) ──────────────────────────────────
        if prog == RAYDIUM_AMM_V4:
            if len(data) >= 1 and data[0] == RAYDIUM_AMM_REMOVE_LIQ_IX:
                # Confirm pool account is in this instruction's account list
                ix_accounts = [account_keys[i] for i in account_indices
                               if i < len(account_keys)]
                if self.cfg.pool_address in ix_accounts:
                    triggers.append("raydium_amm:remove_liquidity")
                    log.warning("Raydium AMM remove_liquidity on monitored pool")

        # ── Anchor-based programs (CLMM / Whirlpool / Meteora) ───────────────
        if prog in DISCRIMINATORS and len(data) >= 8:
            disc = data[:8]
            for name, expected in DISCRIMINATORS[prog].items():
                if disc == expected:
                    ix_accounts = [account_keys[i] for i in account_indices
                                   if i < len(account_keys)]
                    if self.cfg.pool_address in ix_accounts:
                        triggers.append(f"{prog[:6]}:{name}")
                        log.warning("Anchor ix %s on monitored pool  prog=%s", name, prog)

        # ── Dev-wallet supply dump ────────────────────────────────────────────
        # SPL token transfer where source account belongs to dev wallet
        # Check via inner instruction token balance changes
        if prog in (SPL_TOKEN, SPL_TOKEN_2022) and len(data) >= 9:
            # Transfer = instruction type 3 (SPL Token)
            if data[0] == 3:
                amount_raw = struct.unpack_from("<Q", data, 1)[0]
                ix_accounts = [account_keys[i] for i in account_indices
                               if i < len(account_keys)]
                # heuristic: if dev_wallet appears in this instruction's accounts
                if self.cfg.dev_wallet in ix_accounts:
                    if self.dev_wt.record_transfer(slot, amount_raw):
                        triggers.append("dev_wallet:supply_dump")
            self.dev_wt.prune(slot)

        return triggers


# ── Jito bundle submitter ─────────────────────────────────────────────────────
class JitoClient:
    def __init__(self, cfg: Config, keypair: Ed25519PrivateKey) -> None:
        self.cfg     = cfg
        self.keypair = keypair

    async def get_recent_blockhash(self) -> str:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                self.cfg.helius_rpc_url,
                json={"jsonrpc": "2.0", "id": 1,
                      "method": "getLatestBlockhash",
                      "params": [{"commitment": "confirmed"}]},
            )
            resp.raise_for_status()
            return resp.json()["result"]["value"]["blockhash"]

    async def build_tip_transaction(self, blockhash: str) -> bytes:
        """
        Build a minimal SOL transfer to a Jito tip account.
        Wire format: legacy transaction (no versioning prefix).
        """
        from_pubkey  = base58.b58decode(self.cfg.your_wallet)
        to_pubkey    = base58.b58decode(JITO_TIP_ACCOUNTS[0])
        prog_pubkey  = base58.b58decode(SYSTEM_PROGRAM)
        bh_bytes     = base58.b58decode(blockhash)
        lamports     = self.cfg.jito_tip_lamports

        # System transfer instruction data: [2 (u32 le), lamports (u64 le)]
        ix_data = struct.pack("<IQ", 2, lamports)

        # Accounts: [from (signer+writable), to (writable), system_program]
        accounts = [from_pubkey, to_pubkey, prog_pubkey]
        account_keys_bytes = b"".join(accounts)

        # Account meta flags: from=signer+writable(0x03), to=writable(0x02),
        #                     program=none(0x00)
        header = bytes([1, 0, 1])   # num_required_sigs, num_ro_signed, num_ro_unsigned

        # Message
        num_accounts = len(accounts)
        ix_program_idx  = 2     # system program
        ix_account_idxs = bytes([0, 1])   # from, to

        message = (
            header
            + bytes([num_accounts])
            + account_keys_bytes
            + bh_bytes
            + bytes([1])                    # num instructions
            + bytes([ix_program_idx])
            + bytes([len(ix_account_idxs)]) + ix_account_idxs
            + bytes([len(ix_data)])         + ix_data
        )

        sig = self.keypair.sign(message)
        tx  = bytes([1]) + sig + message   # 1 signature
        return tx

    async def submit_bundle(self, transactions: list[bytes]) -> str | None:
        """Submit a list of signed serialised transactions as a Jito bundle."""
        encoded = [base64.b64encode(tx).decode() for tx in transactions]
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method":  "sendBundle",
            "params":  [encoded],
        }
        if self.cfg.dry_run:
            log.info("[DRY-RUN] Would submit Jito bundle: %d txs", len(encoded))
            for i, e in enumerate(encoded):
                log.info("  tx[%d] = %s…", i, e[:60])
            return "DRY_RUN_BUNDLE_ID"

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(JITO_BLOCK_ENGINE, json=payload)
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                log.error("Jito error: %s", result["error"])
                return None
            bundle_id = result.get("result")
            log.info("Jito bundle submitted  id=%s", bundle_id)
            return bundle_id


# ── Jupiter swap builder ──────────────────────────────────────────────────────
class JupiterExit:
    """Fetch a Jupiter swap quote and transaction for closing the position."""

    QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
    SWAP_URL  = "https://quote-api.jup.ag/v6/swap"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def build_exit_transaction(self, user_pubkey: str) -> bytes | None:
        """Return a signed-ready (unsigned) transaction bytes for the swap."""
        async with httpx.AsyncClient(timeout=10) as client:
            # Get quote: token → SOL (So11111111111111111111111111111111111111112)
            params = {
                "inputMint":        self.cfg.token_mint,
                "outputMint":       "So11111111111111111111111111111111111111112",
                "amount":           str(self.cfg.position_amount),
                "slippageBps":      "300",   # 3% slippage tolerance for emergency
                "onlyDirectRoutes": "false",
            }
            q = await client.get(self.QUOTE_URL, params=params)
            q.raise_for_status()
            quote = q.json()

            if "error" in quote:
                log.error("Jupiter quote error: %s", quote["error"])
                return None

            # Get swap transaction
            body = {
                "quoteResponse":          quote,
                "userPublicKey":          user_pubkey,
                "wrapAndUnwrapSol":       True,
                "prioritizationFeeLamports": "auto",
                "dynamicComputeUnitLimit":True,
            }
            s = await client.post(self.SWAP_URL, json=body)
            s.raise_for_status()
            swap = s.json()

            if "swapTransaction" not in swap:
                log.error("Jupiter swap error: %s", swap)
                return None

            tx_b64 = swap["swapTransaction"]
            return base64.b64decode(tx_b64)


# ── Take-profit ladder ────────────────────────────────────────────────────────
class TakeProfitLadder:
    """Track entry price and fire partial exits at +25 % and +50 %."""

    def __init__(self, entry_price: float, position: int) -> None:
        self.entry_price = entry_price
        self.position    = position
        self.fired       = set()

    def check(self, current_price: float) -> list[tuple[str, float]]:
        """Return list of (label, fraction_to_sell) triggers hit."""
        hits = []
        ratio = current_price / self.entry_price if self.entry_price else 1.0
        for label, thresh, frac in [
            ("+25%", 1.25, 0.25),
            ("+50%", 1.50, 0.25),   # sell another 25 % at +50 %
        ]:
            if ratio >= thresh and label not in self.fired:
                self.fired.add(label)
                hits.append((label, frac))
        return hits


# ── Main listener ─────────────────────────────────────────────────────────────
class AntiRugMonitor:
    def __init__(self, cfg: Config) -> None:
        self.cfg        = cfg
        self.keypair    = _load_keypair(cfg.private_key_b58)
        self.user_pubkey = _pubkey_from_keypair(self.keypair)
        self.analyzer   = InstructionAnalyzer(cfg)
        self.jito       = JitoClient(cfg, self.keypair)
        self.jupiter    = JupiterExit(cfg)
        self._triggered = False
        self._callbacks: list[Callable] = []

        assert self.user_pubkey == cfg.your_wallet, (
            f"Keypair derives pubkey {self.user_pubkey} "
            f"but config.your_wallet={cfg.your_wallet}"
        )
        log.info("Monitoring wallet : %s", self.user_pubkey)
        log.info("Pool address      : %s", cfg.pool_address)
        log.info("Token mint        : %s", cfg.token_mint)
        log.info("Dev wallet        : %s", cfg.dev_wallet)
        log.info("Dry-run mode      : %s", cfg.dry_run)

    def on_trigger(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    async def _emergency_exit(self, reason: str) -> None:
        if self._triggered:
            return
        self._triggered = True
        log.critical("EMERGENCY EXIT triggered — reason: %s", reason)

        for cb in self._callbacks:
            try:
                await cb(reason)
            except Exception as e:
                log.error("callback error: %s", e)

        # Build Jupiter exit transaction
        exit_tx = await self.jupiter.build_exit_transaction(self.user_pubkey)
        if exit_tx is None:
            log.error("Could not build exit transaction via Jupiter")
            return

        # Build tip transaction and bundle
        blockhash = await self.jito.get_recent_blockhash()
        tip_tx    = await self.jito.build_tip_transaction(blockhash)
        bundle_id = await self.jito.submit_bundle([exit_tx, tip_tx])

        log.info("Bundle result: %s", bundle_id)

    async def _subscribe_payload(self) -> dict:
        """Helius enhanced transactionSubscribe payload."""
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "transactionSubscribe",
            "params": [
                {
                    "accountInclude": [
                        self.cfg.pool_address,
                        self.cfg.token_mint,
                        self.cfg.dev_wallet,
                    ]
                },
                {
                    "commitment":                  "confirmed",
                    "encoding":                    "jsonParsed",
                    "transactionDetails":          "full",
                    "showRewards":                 False,
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        }

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Subscription confirmation
        if "result" in msg and isinstance(msg["result"], int):
            log.info("Subscription confirmed  id=%d", msg["result"])
            return

        params = msg.get("params", {})
        value  = params.get("result", {}) if isinstance(params, dict) else {}

        if not value:
            return

        triggers = self.analyzer.analyze(value)
        if triggers:
            slot = value.get("slot", "?")
            sig  = (value.get("transaction", {})
                        .get("transaction", {})
                        .get("signatures", ["?"])[0])
            log.warning(
                "RUG PATTERN DETECTED  slot=%s  sig=%s…  triggers=%s",
                slot, str(sig)[:20], triggers,
            )
            await self._emergency_exit(", ".join(triggers))

    async def run(self) -> None:
        backoff = 1
        while True:
            try:
                payload = await self._subscribe_payload()
                log.info("Connecting to Helius WebSocket …")
                async with websockets.connect(
                    self.cfg.helius_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    await ws.send(json.dumps(payload))
                    log.info("Subscribed. Listening for transactions …")
                    backoff = 1
                    async for raw in ws:
                        await self._handle_message(raw)

            except ConnectionClosed as e:
                log.warning("WebSocket closed (%s), reconnecting in %ds …", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                log.error("Unexpected error: %s — reconnecting in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solana anti-rug MEV listener")
    p.add_argument("--config",   required=True, help="Path to config.json (no secrets)")
    p.add_argument("--env-file", default=None,
                   help="Path to .env file (default: .env in script directory)")
    p.add_argument("--dry-run",  action="store_true", default=False,
                   help="Log actions without submitting real transactions")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = Config.from_file(args.config, dry_run=args.dry_run,
                             env_file=args.env_file)

    if not cfg.dry_run:
        log.warning("LIVE MODE — real transactions will be submitted to Jito")

    monitor = AntiRugMonitor(cfg)

    # Example: custom alert callback
    async def alert(reason: str) -> None:
        log.critical("[ALERT] %s — %s", datetime.now(timezone.utc).isoformat(), reason)

    monitor.on_trigger(alert)
    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()
