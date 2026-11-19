// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

interface IERC20 {
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
    function balanceOf(address) external view returns (uint256);
}

/// Block G: a yield vault whose source is NOT verified (no ABI in the cache). Its selectors are in the public
/// signature list, so an agent can recover them from the bytecode. Shares are minted 1:1 and redeemed with a
/// fixed 2% bonus funded at deployment.
contract StakeVault {
    IERC20 public immutable asset;
    mapping(address => uint256) public sharesOf;
    uint256 public totalShares;

    constructor(IERC20 a) {
        asset = a;
    }

    function stakeFor(address who, uint256 amount) external {
        require(asset.transferFrom(msg.sender, address(this), amount), "pull failed");
        sharesOf[who] += amount;
        totalShares += amount;
    }

    function exitShares(uint256 shares, address to) external {
        require(sharesOf[msg.sender] >= shares, "not enough shares");
        sharesOf[msg.sender] -= shares;
        totalShares -= shares;
        require(asset.transfer(to, shares * 102 / 100), "push failed");
    }
}

/// Block G: the implementation behind an EIP-1967 proxy (verified: the cache has its ABI, not the proxy's).
contract SavingsImpl {
    IERC20 public asset;
    mapping(address => uint256) public deposited;
    bool private inited;

    function init(IERC20 a) external {
        require(!inited, "inited");
        inited = true;
        asset = a;
    }

    function lockDeposit(uint256 amount, uint32 lockDays) external {
        require(lockDays >= 1 && lockDays <= 30, "lockDays 1-30");
        require(asset.transferFrom(msg.sender, address(this), amount), "pull failed");
        deposited[msg.sender] += amount;
    }

    function depositOf(address who) external view returns (uint256) {
        return deposited[who];
    }
}

contract Proxy1967 {
    bytes32 private constant SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;

    constructor(address impl, bytes memory initData) {
        assembly {
            sstore(SLOT, impl)
        }
        (bool ok,) = impl.delegatecall(initData);
        require(ok, "init failed");
    }

    fallback() external payable {
        assembly {
            let impl := sload(SLOT)
            calldatacopy(0, 0, calldatasize())
            let r := delegatecall(gas(), impl, 0, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            switch r
            case 0 { revert(0, returndatasize()) }
            default { return(0, returndatasize()) }
        }
    }
}

/// Block G: a constant-product pool with an unfamiliar interface (verified).
contract OddSwap {
    IERC20 public immutable tokenA;
    IERC20 public immutable tokenB;
    uint16 public constant FEE_BPS = 30;

    constructor(IERC20 a, IERC20 b) {
        tokenA = a;
        tokenB = b;
    }

    function previewTrade(address sell, uint256 amountIn) public view returns (uint256) {
        (IERC20 i, IERC20 o) = sell == address(tokenA) ? (tokenA, tokenB) : (tokenB, tokenA);
        require(sell == address(i), "unknown token");
        uint256 x = i.balanceOf(address(this));
        uint256 y = o.balanceOf(address(this));
        uint256 inFee = amountIn * (10000 - FEE_BPS) / 10000;
        return y * inFee / (x + inFee);
    }

    function trade(address sell, uint256 amountIn, uint256 minOut, address to, uint64 deadline) external returns (uint256 out) {
        require(block.timestamp <= deadline, "expired");
        out = previewTrade(sell, amountIn);
        require(out >= minOut, "slippage");
        IERC20 o = sell == address(tokenA) ? tokenB : tokenA;
        require(IERC20(sell).transferFrom(msg.sender, address(this), amountIn), "pull failed");
        require(o.transfer(to, out), "push failed");
    }
}
