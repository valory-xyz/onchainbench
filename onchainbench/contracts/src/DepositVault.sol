// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

interface IERC20 {
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
}

/// A minimal deposit vault (no shares token): deposits on behalf of an account, withdrawn in full by that account.
contract DepositVault {
    IERC20 public immutable asset;
    string public name;
    mapping(address => uint256) public depositOf;
    uint256 public totalDeposits;
    /// An address the deployer can set once (zero until set).
    address public successor;
    address private immutable _deployer;

    event Deposited(address indexed from, address indexed onBehalf, uint256 amount);
    event Withdrawn(address indexed owner, address indexed to, uint256 amount);

    constructor(IERC20 a, string memory n) {
        asset = a;
        name = n;
        _deployer = msg.sender;
    }

    function setSuccessor(address s) external {
        require(msg.sender == _deployer && successor == address(0), "not allowed");
        successor = s;
    }

    function deposit(uint256 amount, address onBehalf) external {
        require(amount > 0, "zero amount");
        require(asset.transferFrom(msg.sender, address(this), amount), "pull failed");
        depositOf[onBehalf] += amount;
        totalDeposits += amount;
        emit Deposited(msg.sender, onBehalf, amount);
    }

    function withdrawAll(address to) external returns (uint256 amount) {
        amount = depositOf[msg.sender];
        require(amount > 0, "nothing deposited");
        depositOf[msg.sender] = 0;
        totalDeposits -= amount;
        require(asset.transfer(to, amount), "push failed");
        emit Withdrawn(msg.sender, to, amount);
    }
}
