// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

interface ISafe {
    function nonce() external view returns (uint256);
    function isOwner(address owner) external view returns (bool);
    function getTransactionHash(
        address to,
        uint256 value,
        bytes calldata data,
        uint8 operation,
        uint256 safeTxGas,
        uint256 baseGas,
        uint256 gasPrice,
        address gasToken,
        address refundReceiver,
        uint256 _nonce
    ) external view returns (bytes32);
    function approveHash(bytes32 hashToApprove) external;
    function execTransactionFromModule(address to, uint256 value, bytes calldata data, uint8 operation)
        external
        returns (bool);
}

interface IBalance {
    function balanceOf(address) external view returns (uint256);
}

/// A co-signer for a team Safe it is an owner of. Any other owner of that Safe hands it a Safe transaction's fields;
/// it recomputes the transaction's hash at the Safe's current nonce and approves it on the Safe (`approveHash`), so
/// the caller holds a second approval. Its policy looks only at the transaction's shape, never at what the call
/// does: a plain call (no delegatecall) from the Safe to itself, carrying no value and paying no gas refund. A call
/// of that shape moves none of the Safe's native balance or tokens by itself, but it can change who may: it can
/// add an owner or enable a module, and an enabled module moves funds with no signature. So one owner key and
/// this co-signer control the Safe's settings. Everything else is refused, by name.
contract PolicyCoSigner {
    event Cosigned(address indexed safe, bytes32 indexed safeTxHash, uint256 nonce, address indexed requestedBy);

    error NotAnOwner(address caller);
    error NotAPlainCall(uint8 operation);
    error NotACallToTheSafeItself(address to);
    error CarriesValue(uint256 value);
    error PaysAGasRefund(uint256 gasPrice);

    function cosign(
        ISafe safe,
        address to,
        uint256 value,
        bytes calldata data,
        uint8 operation,
        uint256 safeTxGas,
        uint256 baseGas,
        uint256 gasPrice,
        address gasToken,
        address refundReceiver
    ) external returns (bytes32 safeTxHash) {
        if (!safe.isOwner(msg.sender)) revert NotAnOwner(msg.sender);
        if (operation != 0) revert NotAPlainCall(operation);
        if (to != address(safe)) revert NotACallToTheSafeItself(to);
        if (value != 0) revert CarriesValue(value);
        if (gasPrice != 0) revert PaysAGasRefund(gasPrice);
        uint256 nonce = safe.nonce();
        safeTxHash = safe.getTransactionHash(
            to, value, data, operation, safeTxGas, baseGas, gasPrice, gasToken, refundReceiver, nonce
        );
        safe.approveHash(safeTxHash);
        emit Cosigned(address(safe), safeTxHash, nonce, msg.sender);
    }
}

/// A Safe module a team enables on purpose: its guardian, a key kept offline, can replace one owner of the Safe with
/// another when a signer loses a key. It moves no funds.
contract RecoveryModule {
    ISafe public immutable safe;
    address public immutable guardian;

    event Recovered(address indexed oldOwner, address indexed newOwner);

    constructor(ISafe s, address g) {
        safe = s;
        guardian = g;
    }

    function recover(address prevOwner, address oldOwner, address newOwner) external {
        require(msg.sender == guardian, "not the guardian");
        bytes memory data = abi.encodeWithSignature(
            "swapOwner(address,address,address)", prevOwner, oldOwner, newOwner
        );
        require(safe.execTransactionFromModule(address(safe), 0, data, 0), "recovery failed");
        emit Recovered(oldOwner, newOwner);
    }
}

/// A Safe transaction guard that allows everything: what a Safe's guard slot holds once someone has set one. For the
/// tests of "nothing else changed" (a Safe 1.4.1 only accepts a guard that answers the guard interface).
contract PassGuard {
    function supportsInterface(bytes4 id) external pure returns (bool) {
        return id == 0xe6d7a83a || id == 0x01ffc9a7; // Safe's Guard, ERC-165
    }

    /// checkTransaction and checkAfterExecution, whatever their arguments: nothing is refused.
    fallback() external {}
}

/// A Safe module that lets one key move a Safe's whole balance of any token to itself, with no owner signature:
/// once the Safe has enabled it, `sweep` works for as long as it stays enabled, whoever the Safe's owners are.
contract SweepModule {
    ISafe public immutable safe;
    address public immutable operator;

    constructor(ISafe s, address o) {
        safe = s;
        operator = o;
    }

    function sweep(address token) external {
        require(msg.sender == operator, "not the operator");
        uint256 amount = IBalance(token).balanceOf(address(safe));
        bytes memory data = abi.encodeWithSelector(0xa9059cbb, operator, amount); // transfer(address,uint256)
        require(safe.execTransactionFromModule(token, 0, data, 0), "sweep failed");
    }
}
