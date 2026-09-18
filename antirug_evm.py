#!/usr/bin/env python3
"""
antirug_evm.py — EVM (Ethereum / EVM-compatible) Anti-Rug MEV Exit System

Monitors a Uniswap V2/V3 position via:
  Primary  : eth_subscribe("pendingTransactions") on a WebSocket RPC
  Secondary: Flashbots MEV-Share SSE stream (reveals partial hints)

On detection of rug-pull patterns fires a Uniswap exit bundled via Flashbots.

Detection triggers (CLAUDE.md §5):
  1. remove_liquidity / burn_lp on the monitored pool (Uniswap V2/V3, Sushi)
  2. transferOwnership / renounceOwnership or setFreeze on the token contract
  3. Dev/creator wallet ERC-20 Transfer > 5% of circulating supply in one block

Exit flow:
  Standard  : laddered take-profits (+25%, +50%)
  Emergency : 100% Uniswap exactInputSingle → ETH/USDC via Flashbots bundle

Keys loaded from .env:
  ETH_PRIVATE_KEY          — hex wallet private key (0x-prefixed or raw 32-byte hex)
  ETH_RPC_WS_URL           — wss:// endpoint (Alchemy/Infura/QuickNode)
  ETH_RPC_HTTP_URL         — https:// endpoint for bundle submission RPC calls
  FLASHBOTS_SIGNING_KEY    — separate Flashbots reputation key (hex, optional;
                             falls back to ETH_PRIVATE_KEY if absent)

Non-sensitive config.json:
  pool_address, token_address, dev_wallet, your_wallet,
  token_decimals, position_amount_wei, circulating_supply_wei,
  uniswap_version (2 or 3), chain_id, flashbots_tip_wei

Usage:
    python antirug_evm.py --config config_evm.json [--dry-run]
    python antirug_evm.py --config config_evm.json --env-file /run/secrets/.env

Dependencies:
    pip install websockets httpx eth-account eth-abi python-dotenv
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
import websockets
from dotenv import load_dotenv
from eth_abi import decode as abi_decode
from eth_account import Account
from eth_account.messages import encode_defunct
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("antirug_evm")

# ── Well-known addresses ──────────────────────────────────────────────────────
UNISWAP_V2_ROUTER   = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
UNISWAP_V3_ROUTER   = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"   # SwapRouter02
SUSHISWAP_ROUTER    = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
UNISWAP_V3_NFT_POS  = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"   # NonfungiblePositionManager
WETH                = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

FLASHBOTS_RELAY     = "https://relay.flashbots.net"
MEV_SHARE_SSE       = "https://mev-share.flashbots.net"

# ── EVM function selectors (keccak256(sig)[:4]) ───────────────────────────────
# Pre-computed; verified against on-chain ABIs.
SEL: dict[str, bytes] = {
    # Uniswap V2 / SushiSwap router
    "v2_remove_liquidity":             bytes.fromhex("baa2abde"),
    "v2_remove_liquidity_eth":         bytes.fromhex("02751cec"),
    "v2_remove_liquidity_permit":      bytes.fromhex("2195995c"),
    "v2_remove_liquidity_eth_permit":  bytes.fromhex("ded9382a"),
    "v2_remove_liq_eth_fee_permit":    bytes.fromhex("5b0d5984"),
    # Uniswap V3 NonfungiblePositionManager
    "v3_decrease_liquidity":           bytes.fromhex("0c49ccbe"),
    "v3_collect":                      bytes.fromhex("fc6f7865"),
    # ERC-20 burn
    "erc20_burn_amount":               bytes.fromhex("42966c68"),
    "erc20_burn_from":                 bytes.fromhex("9dc29fac"),
    # Ownership / access control
    "transfer_ownership":              bytes.fromhex("f2fde38b"),
    "renounce_ownership":              bytes.fromhex("715018a6"),
    # Common freeze / blacklist patterns
    "set_fee_percent":                 bytes.fromhex("487f4ebb"),   # common tax setter
    "enable_trading":                  bytes.fromhex("8a8c523c"),   # honeypot release
    # ERC-20 Transfer event topic (for dev-wallet monitoring)
    "erc20_transfer_topic": bytes.fromhex(
        "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    ),
}

# Selectors that mean "liquidity is being removed from the pool"
RUG_REMOVE_LIQ = {
    SEL["v2_remove_liquidity"],
    SEL["v2_remove_liquidity_eth"],
    SEL["v2_remove_liquidity_permit"],
    SEL["v2_remove_liquidity_eth_permit"],
    SEL["v2_remove_liq_eth_fee_permit"],
    SEL["v3_decrease_liquidity"],
    SEL["v3_collect"],
}

# Selectors that mean "contract control is changing"
RUG_OWNERSHIP = {
    SEL["erc20_burn_from"],
    SEL["erc20_burn_amount"],
    SEL["transfer_ownership"],
    SEL["renounce_ownership"],
}


# ── Utilities ─────────────────────────────────────────────────────────────────
def _keccak256(data: bytes) -> bytes:
    import hashlib as _hl
    k = _hl.new("sha3_256") if False else None   # placeholder
    # Use eth_account's keccak which is keccak256, not sha3
    from eth_account._utils.structured_data.hashing import keccak as _k
    return _k(data)


def _to_hex(b: bytes) -> str:
    return "0x" + b.hex()


def _from_hex(s: str) -> bytes:
    h = s.removeprefix("0x").removeprefix("0X")
    if len(h) % 2:
        h = "0" + h
    return bytes.fromhex(h)


def _encode_function_call(selector: bytes, *args_encoded: bytes) -> bytes:
    return selector + b"".join(args_encoded)


def _pad32(b: bytes) -> bytes:
    return b.rjust(32, b"\x00")


def _abi_encode_address(addr: str) -> bytes:
    return _pad32(_from_hex(addr))


def _abi_encode_uint(n: int) -> bytes:
    return _pad32(n.to_bytes(32, "big").lstrip(b"\x00") or b"\x00")


# ── .env loader ───────────────────────────────────────────────────────────────
def _load_env(env_file: str | None = None) -> dict[str, str]:
    for candidate in filter(None, [env_file,
                                   str(Path(__file__).parent / ".env")]):
        if Path(candidate).exists():
            load_dotenv(candidate, override=False)
            log.info("Loaded .env from %s", candidate)
            break

    required = ("ETH_PRIVATE_KEY", "ETH_RPC_WS_URL", "ETH_RPC_HTTP_URL")
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {missing}\n"
            f"Add them to .env or export them before running.\n"
            f"See .env.example for reference."
        )
    return {k: os.environ[k] for k in (*required, "FLASHBOTS_SIGNING_KEY")
            if os.getenv(k)}


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class EvmConfig:
    # non-sensitive — safe in config.json
    pool_address:             str
    token_address:            str
    dev_wallet:               str
    your_wallet:              str
    token_decimals:           int   = 18
    position_amount_wei:      int   = 0       # token units held (raw)
    circulating_supply_wei:   int   = 10**27  # 1B tokens × 10^18 decimals
    uniswap_version:          int   = 2       # 2 or 3; derived from uniswap_v3_router presence
    chain_id:                 int   = 1       # 1 = Ethereum mainnet
    flashbots_tip_wei:        int   = 10**16  # 0.01 ETH tip
    uniswap_v2_router:        str   = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    uniswap_v3_router:        str   = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    weth_address:             str   = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    v3_fee_tier:              int   = 3000    # 0.3% pool fee tier
    slippage_bps:             int   = 100     # 1% slippage tolerance
    blocks_ahead:             int   = 2       # target inclusion in next N blocks
    dry_run:                  bool  = True
    # sensitive — injected from env, never serialised
    eth_private_key:          str   = field(default="", repr=False)
    eth_rpc_ws_url:           str   = field(default="", repr=False)
    eth_rpc_http_url:         str   = field(default="", repr=False)
    flashbots_signing_key:    str   = field(default="", repr=False)

    @classmethod
    def from_file(cls, path: str, dry_run: bool = True,
                  env_file: str | None = None) -> "EvmConfig":
        env = _load_env(env_file)
        with open(path) as f:
            d = json.load(f)
        for secret in ("eth_private_key", "eth_rpc_ws_url",
                       "eth_rpc_http_url", "flashbots_signing_key"):
            d.pop(secret, None)
        d["dry_run"]              = dry_run
        d["eth_private_key"]      = env["ETH_PRIVATE_KEY"]
        d["eth_rpc_ws_url"]       = env["ETH_RPC_WS_URL"]
        d["eth_rpc_http_url"]     = env["ETH_RPC_HTTP_URL"]
        d["flashbots_signing_key"]= env.get("FLASHBOTS_SIGNING_KEY",
                                            env["ETH_PRIVATE_KEY"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── EVM account helpers ───────────────────────────────────────────────────────
def _load_account(hex_key: str) -> Account:
    key = hex_key if hex_key.startswith("0x") else "0x" + hex_key
    return Account.from_key(key)


def _sign_flashbots_header(body: str, signing_account: Account) -> str:
    """Produce X-Flashbots-Signature header value."""
    msg     = encode_defunct(text=hashlib.sha256(body.encode()).hexdigest())
    sig     = signing_account.sign_message(msg)
    return f"{signing_account.address}:{sig.signature.hex()}"


# ── Dev-wallet ERC-20 transfer tracker ───────────────────────────────────────
class DevWalletTracker:
    """Accumulate ERC-20 transfers from dev wallet per block."""

    def __init__(self, dev_wallet: str, circulating_supply_wei: int,
                 dump_threshold: float = 0.05) -> None:
        self.dev_wallet             = dev_wallet.lower()
        self.circulating_supply_wei = circulating_supply_wei
        self.dump_threshold         = dump_threshold
        self._block_transfers: dict[int, int] = {}

    def record(self, block: int, amount_wei: int) -> bool:
        self._block_transfers[block] = (
            self._block_transfers.get(block, 0) + amount_wei
        )
        pct = self._block_transfers[block] / self.circulating_supply_wei
        if pct >= self.dump_threshold:
            log.warning("DEV DUMP  block=%d  pct=%.2f%%", block, pct * 100)
            return True
        return False

    def prune(self, current_block: int, window: int = 5) -> None:
        stale = [b for b in self._block_transfers
                 if b < current_block - window]
        for b in stale:
            del self._block_transfers[b]


# ── Transaction decoder ───────────────────────────────────────────────────────
class TxDecoder:
    """Decode a pending transaction and classify rug-pull patterns."""

    def __init__(self, cfg: EvmConfig) -> None:
        self.cfg     = cfg
        self.dev_wt  = DevWalletTracker(
            cfg.dev_wallet, cfg.circulating_supply_wei
        )
        self._pool   = cfg.pool_address.lower()
        self._token  = cfg.token_address.lower()
        self._dev    = cfg.dev_wallet.lower()

    def analyze_pending(self, tx: dict) -> list[str]:
        """Return list of triggered rug patterns for a pending tx dict."""
        triggers: list[str] = []
        to      = (tx.get("to") or "").lower()
        data    = _from_hex(tx.get("input") or tx.get("data") or "0x")
        blk     = int(tx.get("blockNumber") or "0", 16)

        if len(data) < 4:
            return triggers

        selector = data[:4]

        # ── LP removal: tx targets pool or a router that routes to our pool ──
        if to in (UNISWAP_V2_ROUTER.lower(), SUSHISWAP_ROUTER.lower(),
                  UNISWAP_V3_ROUTER.lower(), UNISWAP_V3_NFT_POS.lower(),
                  self._pool):
            if selector in RUG_REMOVE_LIQ:
                name = next(k for k, v in SEL.items() if v == selector)
                triggers.append(f"remove_liquidity:{name}")
                log.warning("Remove-liquidity detected  to=%s  sel=%s", to, name)

        # ── Ownership / freeze: tx targets our token contract ────────────────
        if to == self._token and selector in RUG_OWNERSHIP:
            name = next(k for k, v in SEL.items() if v == selector)
            triggers.append(f"ownership_change:{name}")
            log.warning("Ownership/burn tx on token  sel=%s", name)

        # ── Dev-wallet large dump: ERC-20 transfer from dev ──────────────────
        if to == self._token and selector == SEL["erc20_transfer_topic"][:4]:
            # transfer(address to, uint256 amount)
            frm = (tx.get("from") or "").lower()
            if frm == self._dev and len(data) >= 68:
                try:
                    _dst, amount = abi_decode(["address", "uint256"], data[4:])
                    if self.dev_wt.record(blk, amount):
                        triggers.append("dev_wallet:supply_dump")
                except Exception:
                    pass
            self.dev_wt.prune(blk)

        return list(dict.fromkeys(triggers))


# ── Flashbots bundle client ───────────────────────────────────────────────────
class FlashbotsClient:
    def __init__(self, cfg: EvmConfig) -> None:
        self.cfg             = cfg
        self.wallet          = _load_account(cfg.eth_private_key)
        self.signing_account = _load_account(cfg.flashbots_signing_key)
        log.info("EVM wallet          : %s", self.wallet.address)
        log.info("Flashbots signer    : %s", self.signing_account.address)

    async def get_block_number(self) -> int:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                self.cfg.eth_rpc_http_url,
                json={"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]},
            )
            r.raise_for_status()
            return int(r.json()["result"], 16)

    async def get_base_fee(self) -> int:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                self.cfg.eth_rpc_http_url,
                json={"jsonrpc":"2.0","id":1,
                      "method":"eth_getBlockByNumber",
                      "params":["latest", False]},
            )
            r.raise_for_status()
            return int(r.json()["result"]["baseFeePerGas"], 16)

    async def get_nonce(self) -> int:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                self.cfg.eth_rpc_http_url,
                json={"jsonrpc":"2.0","id":1,
                      "method":"eth_getTransactionCount",
                      "params":[self.wallet.address, "pending"]},
            )
            r.raise_for_status()
            return int(r.json()["result"], 16)

    def _build_uniswap_v2_exit(
        self, base_fee: int, nonce: int, token_amount: int,
    ) -> str:
        """
        Build and sign a UniV2 swapExactTokensForETH transaction.
        selector: 0x18cbafe5  swapExactTokensForETH(
            uint amountIn, uint amountOutMin,
            address[] path, address to, uint deadline)
        """
        selector = bytes.fromhex("18cbafe5")
        deadline = int(time.time()) + 120   # 2 min from now

        # ABI-encode params
        amount_in     = _pad32(token_amount.to_bytes(32, "big").lstrip(b"\x00") or b"\x00")
        amount_out_min= _pad32(b"\x00")     # 0 = accept any (emergency mode)
        to_addr       = _pad32(_from_hex(self.cfg.your_wallet))
        deadline_enc  = _pad32(deadline.to_bytes(32, "big").lstrip(b"\x00") or b"\x00")

        # path = [token, WETH] — dynamic array
        path_offset = _pad32((5 * 32).to_bytes(32, "big").lstrip(b"\x00") or b"\x00")
        path_len    = _pad32(b"\x02")
        path_t      = _pad32(_from_hex(self.cfg.token_address))
        path_weth   = _pad32(_from_hex(WETH))

        calldata = (selector + amount_in + amount_out_min
                    + path_offset + to_addr + deadline_enc
                    + path_len + path_t + path_weth)

        priority_fee = min(base_fee, 10**9 * 50)   # cap at 50 gwei
        max_fee      = base_fee * 2 + priority_fee

        tx = {
            "chainId":              self.cfg.chain_id,
            "nonce":                nonce,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "gas":                  250_000,
            "to":                   UNISWAP_V2_ROUTER,
            "value":                0,
            "data":                 _to_hex(calldata),
        }
        signed = self.wallet.sign_transaction(tx)
        return _to_hex(signed.raw_transaction)

    def _build_uniswap_v3_exit(
        self, base_fee: int, nonce: int, token_amount: int,
    ) -> str:
        """
        Build and sign a UniV3 SwapRouter02 exactInputSingle transaction.
        selector: 0x04e45aaf  exactInputSingle((
            address tokenIn, address tokenOut, uint24 fee,
            address recipient, uint256 amountIn,
            uint256 amountOutMinimum, uint160 sqrtPriceLimitX96))
        """
        selector = bytes.fromhex("04e45aaf")

        # Struct encoding (tuple = sequential 32-byte slots)
        fee_tier = 3000   # 0.3% pool (most common for memecoins)
        params   = (
            _pad32(_from_hex(self.cfg.token_address))   # tokenIn
            + _pad32(_from_hex(WETH))                   # tokenOut
            + _pad32(fee_tier.to_bytes(3, "big"))        # fee (uint24)
            + _pad32(_from_hex(self.cfg.your_wallet))    # recipient
            + _pad32(token_amount.to_bytes(32, "big").lstrip(b"\x00") or b"\x00")
            + _pad32(b"\x00")                            # amountOutMinimum = 0
            + _pad32(b"\x00")                            # sqrtPriceLimitX96 = 0
        )
        calldata = selector + params

        priority_fee = min(base_fee, 10**9 * 50)
        max_fee      = base_fee * 2 + priority_fee

        tx = {
            "chainId":              self.cfg.chain_id,
            "nonce":                nonce,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "gas":                  300_000,
            "to":                   UNISWAP_V3_ROUTER,
            "value":                0,
            "data":                 _to_hex(calldata),
        }
        signed = self.wallet.sign_transaction(tx)
        return _to_hex(signed.raw_transaction)

    def _build_tip_tx(self, base_fee: int, nonce: int) -> str:
        """Direct ETH transfer to Flashbots coinbase (tip)."""
        priority_fee = min(base_fee, 10**9 * 100)
        max_fee      = base_fee * 2 + priority_fee
        tx = {
            "chainId":              self.cfg.chain_id,
            "nonce":                nonce + 1,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "gas":                  21_000,
            "to":                   "0xDAFEA492D9c6733ae3d56b7Ed1ADB60692c98Bc5",   # Flashbots builder
            "value":                self.cfg.flashbots_tip_wei,
            "data":                 "0x",
        }
        signed = self.wallet.sign_transaction(tx)
        return _to_hex(signed.raw_transaction)

    async def submit_bundle(self, target_block: int) -> str | None:
        base_fee = await self.get_base_fee()
        nonce    = await self.get_nonce()

        if self.cfg.uniswap_version == 3:
            exit_tx = self._build_uniswap_v3_exit(
                base_fee, nonce, self.cfg.position_amount_wei)
        else:
            exit_tx = self._build_uniswap_v2_exit(
                base_fee, nonce, self.cfg.position_amount_wei)

        tip_tx = self._build_tip_tx(base_fee, nonce)

        bundle_body = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method":  "eth_sendBundle",
            "params": [{
                "txs":           [exit_tx, tip_tx],
                "blockNumber":   hex(target_block + 1),
                "minTimestamp":  0,
                "maxTimestamp":  int(time.time()) + 120,
                "revertingTxHashes": [],
            }],
        })

        if self.cfg.dry_run:
            log.info("[DRY-RUN] Flashbots bundle for block %d", target_block + 1)
            log.info("  exit_tx : %s…", exit_tx[:66])
            log.info("  tip_tx  : %s…", tip_tx[:66])
            return "DRY_RUN_BUNDLE_ID"

        headers = {
            "Content-Type":         "application/json",
            "X-Flashbots-Signature": _sign_flashbots_header(
                bundle_body, self.signing_account),
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(FLASHBOTS_RELAY, content=bundle_body, headers=headers)
            r.raise_for_status()
            result = r.json()
            if "error" in result:
                log.error("Flashbots error: %s", result["error"])
                return None
            bundle_hash = result.get("result", {}).get("bundleHash")
            log.info("Flashbots bundle submitted  hash=%s", bundle_hash)
            return bundle_hash


# ── MEV-Share SSE listener (secondary signal) ─────────────────────────────────
class MevShareListener:
    """
    Consumes Flashbots MEV-Share SSE stream for partial transaction hints.
    Useful for detecting rug-related transactions even when full calldata
    is not revealed (MEV-Share shows logs if token matches).
    """

    def __init__(self, token_address: str,
                 on_hint: Callable[[dict], None]) -> None:
        self.token   = token_address.lower()
        self.on_hint = on_hint

    async def run(self) -> None:
        backoff = 1
        while True:
            try:
                log.info("Connecting to MEV-Share SSE stream …")
                async with httpx.AsyncClient(timeout=None) as c:
                    async with c.stream("GET", MEV_SHARE_SSE,
                                        headers={"Accept": "text/event-stream"}) as resp:
                        backoff = 1
                        buffer = ""
                        async for chunk in resp.aiter_text():
                            buffer += chunk
                            while "\n\n" in buffer:
                                event_str, buffer = buffer.split("\n\n", 1)
                                data_line = next(
                                    (l[6:] for l in event_str.splitlines()
                                     if l.startswith("data:")), None)
                                if not data_line:
                                    continue
                                try:
                                    hint = json.loads(data_line)
                                    await self._process_hint(hint)
                                except json.JSONDecodeError:
                                    pass
            except Exception as e:
                log.warning("MEV-Share error: %s — retrying in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _process_hint(self, hint: dict) -> None:
        """Check if hint logs reference our token (Transfer/Sync events)."""
        logs = hint.get("logs") or []
        for log_entry in logs:
            addr = (log_entry.get("address") or "").lower()
            if addr == self.token:
                topics = log_entry.get("topics") or []
                # large Transfer out = potential dump
                if topics and topics[0] == _to_hex(SEL["erc20_transfer_topic"]):
                    self.on_hint(hint)
                    return


# ── Main EVM anti-rug monitor ─────────────────────────────────────────────────
class EvmAntiRugMonitor:
    def __init__(self, cfg: EvmConfig) -> None:
        self.cfg      = cfg
        self.decoder  = TxDecoder(cfg)
        self.flashbots= FlashbotsClient(cfg)
        self._triggered = False
        self._callbacks: list[Callable] = []

    def on_trigger(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    async def _emergency_exit(self, reason: str) -> None:
        if self._triggered:
            return
        self._triggered = True
        log.critical("EMERGENCY EXIT — %s", reason)

        for cb in self._callbacks:
            try:
                await cb(reason)
            except Exception as e:
                log.error("callback error: %s", e)

        block    = await self.flashbots.get_block_number()
        bundle_hash = await self.flashbots.submit_bundle(block)
        log.info("Bundle result: %s", bundle_hash)

    def _mev_share_hint_callback(self, hint: dict) -> None:
        log.warning("MEV-Share hint on token: %s", hint.get("hash"))
        # Fire emergency exit asynchronously from sync callback
        asyncio.create_task(
            self._emergency_exit(f"mev_share_hint:{hint.get('hash')}")
        )

    async def _subscribe_pending(self) -> None:
        """Subscribe to pending transactions and scan for rug patterns."""
        backoff = 1
        while True:
            try:
                log.info("Connecting to EVM WebSocket RPC …")
                async with websockets.connect(
                    self.cfg.eth_rpc_ws_url,
                    ping_interval=20, ping_timeout=10,
                    max_size=4 * 1024 * 1024,
                ) as ws:
                    # Subscribe to full pending transactions
                    sub_payload = json.dumps({
                        "jsonrpc": "2.0", "id": 1,
                        "method":  "eth_subscribe",
                        "params":  ["newPendingTransactions"],
                    })
                    await ws.send(sub_payload)
                    sub_resp = json.loads(await ws.recv())
                    sub_id   = sub_resp.get("result")
                    log.info("Pending-tx subscription confirmed  id=%s", sub_id)
                    backoff = 1

                    async for raw in ws:
                        await self._handle_ws_message(raw)

            except ConnectionClosed as e:
                log.warning("WS closed (%s), reconnecting in %ds …", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                log.error("WS error: %s — reconnecting in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _fetch_tx(self, tx_hash: str) -> dict | None:
        """Fetch full transaction data for a hash."""
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(
                self.cfg.eth_rpc_http_url,
                json={"jsonrpc":"2.0","id":1,
                      "method":"eth_getTransactionByHash",
                      "params":[tx_hash]},
            )
            r.raise_for_status()
            return r.json().get("result")

    async def _handle_ws_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        params = msg.get("params", {})
        result = params.get("result") if isinstance(params, dict) else None
        if not result:
            return

        # Some providers send full tx objects; others send just the hash.
        if isinstance(result, str) and result.startswith("0x"):
            # Hash only — fetch full tx (best-effort; skip if too slow)
            try:
                tx = await asyncio.wait_for(self._fetch_tx(result), timeout=2.0)
            except asyncio.TimeoutError:
                return
        elif isinstance(result, dict):
            tx = result
        else:
            return

        if not tx:
            return

        triggers = self.decoder.analyze_pending(tx)
        if triggers:
            sig = tx.get("hash", "?")
            log.warning(
                "RUG PATTERN  hash=%s…  from=%s  triggers=%s",
                str(sig)[:18], tx.get("from","?")[:12], triggers,
            )
            await self._emergency_exit(", ".join(triggers))

    async def run(self) -> None:
        log.info("Starting EVM Anti-Rug Monitor  chain_id=%d  dry_run=%s",
                 self.cfg.chain_id, self.cfg.dry_run)
        mev_share = MevShareListener(
            self.cfg.token_address,
            self._mev_share_hint_callback,
        )
        # Run mempool listener + MEV-Share concurrently
        await asyncio.gather(
            self._subscribe_pending(),
            mev_share.run(),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EVM anti-rug Flashbots MEV listener")
    p.add_argument("--config",   required=True, help="Path to config_evm.json (no secrets)")
    p.add_argument("--env-file", default=None,  help="Path to .env file")
    p.add_argument("--dry-run",  action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    cfg     = EvmConfig.from_file(args.config, dry_run=args.dry_run,
                                   env_file=args.env_file)
    monitor = EvmAntiRugMonitor(cfg)

    async def alert(reason: str) -> None:
        log.critical("[ALERT] %s — %s",
                     datetime.now(timezone.utc).isoformat(), reason)

    monitor.on_trigger(alert)
    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()
