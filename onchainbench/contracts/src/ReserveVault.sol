// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

interface IERC20 {
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
}

/// A reserve vault with one owner. Anyone may deposit for a receiver. An account withdraws its own deposit; the
/// owner may withdraw any account's deposit, and may hand the vault over to a new owner.
contract ReserveVault {
    IERC20 public immutable asset;
    string public name;
    address public owner;
    mapping(address => uint256) public depositOf;
    uint256 public totalAssets;

    event Deposited(address indexed sender, address indexed receiver, uint256 assets);
    event Withdrawn(address indexed caller, address indexed receiver, address indexed account, uint256 assets);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(IERC20 a, string memory n, address o) {
        asset = a;
        name = n;
        owner = o;
        emit OwnershipTransferred(address(0), o);
    }

    function deposit(uint256 assets, address receiver) external returns (uint256) {
        require(assets > 0, "zero amount");
        require(asset.transferFrom(msg.sender, address(this), assets), "pull failed");
        depositOf[receiver] += assets;
        totalAssets += assets;
        emit Deposited(msg.sender, receiver, assets);
        return assets;
    }

    function withdraw(uint256 assets, address receiver, address account) external returns (uint256) {
        require(msg.sender == account || msg.sender == owner, "not the account or the owner");
        require(depositOf[account] >= assets, "more than deposited");
        depositOf[account] -= assets;
        totalAssets -= assets;
        require(asset.transfer(receiver, assets), "push failed");
        emit Withdrawn(msg.sender, receiver, account, assets);
        return assets;
    }

    function transferOwnership(address newOwner) external {
        require(msg.sender == owner, "not the owner");
        require(newOwner != address(0), "zero owner");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}
