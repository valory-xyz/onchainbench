"""Chain data for the 9 chains the bench covers. Data only (the benchmark's copy): prices for the broker's USD caps.

- **The 9 chains** (2026-11-09): Ethereum, Gnosis, Polygon, Arbitrum, Optimism, Base and Celo (the
  ServiceManager upgrade of June 2026), plus Mode (still configured by Pearl and Connect) and Robinhood
  Chain (deployment approved in Q3 2026).
- **Any other EVM chain** works once the user supplies its native token's USD price
  (`--native-usd <chain>=<price>`); without one, value calls there are refused with that instruction.
- **Mech payment assets** come from the mech itself (Connect's `mech_tools` reports `payment_token`), so
  every mech request names its mech. The table's `facilitator_asset` is what the mech
  facilitator charges in on each chain it serves; it goes away with T7.
- **Prices are rough** and only make USD session caps comparable across chains. They're a stopgap,
  replaced by Connect's scoped-token caps (T4) once Connect ships them. Connect's own supported-chains list (T8)
  would replace this table.
"""

from __future__ import annotations

from dataclasses import dataclass

NATIVE = "native"
STABLE6 = "stable6"  # a 6-decimal USD stablecoin (USDC on most chains, USDG on Robinhood)


@dataclass(frozen=True)
class ChainInfo:
    native_symbol: str
    native_usd: float  # rough
    facilitator_asset: str | None  # NATIVE, STABLE6, or None (the facilitator doesn't serve this chain)


OLAS_EVM_CHAINS: dict[str, ChainInfo] = {
    "ethereum": ChainInfo("ETH", 3000.0, None),
    "gnosis": ChainInfo("xDAI", 1.0, NATIVE),
    "polygon": ChainInfo("POL", 0.25, STABLE6),
    "arbitrum": ChainInfo("ETH", 3000.0, None),
    "optimism": ChainInfo("ETH", 3000.0, STABLE6),
    "base": ChainInfo("ETH", 3000.0, STABLE6),
    "celo": ChainInfo("CELO", 0.5, None),
    "mode": ChainInfo("ETH", 3000.0, None),
    "robinhood": ChainInfo("ETH", 3000.0, None),
}

# Tokens counted at $1 when a mech is paid in them (matched on the on-chain `symbol()`).
STABLE_SYMBOLS = {"USDC", "USDC.E", "USDBC", "USDT", "USDT0", "DAI", "XDAI", "WXDAI", "USDG", "USDS"}


def native_usd(chain: str | None, overrides: dict[str, float] | None = None) -> float | None:
    """The native token's USD price: the user's override first, then the table; None if unknown."""
    if chain and overrides and chain in overrides:
        return overrides[chain]
    info = OLAS_EVM_CHAINS.get(chain or "")
    return info.native_usd if info else None
