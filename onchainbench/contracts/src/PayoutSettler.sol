// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

interface IERC20 {
    function transferFrom(address, address, uint256) external returns (bool);
}

/// A token payout settler: settles a batch of legs (payee and amount) from the caller's balance under a memo.
contract PayoutSettler {
    IERC20 public immutable token;
    mapping(bytes32 => bool) public settledMemo;

    event Settled(bytes32 indexed memo, address indexed to, uint256 amount);

    constructor(IERC20 t) {
        token = t;
    }

    function settle(bytes32[] calldata legs, bytes32 memo) external returns (uint256 total) {
        require(memo != bytes32(0), "memo required");
        require(!settledMemo[memo], "memo already settled");
        require(legs.length > 0, "no legs");
        settledMemo[memo] = true;
        for (uint256 i = 0; i < legs.length; i++) {
            address to = address(uint160(uint256(legs[i]) >> 96));
            uint256 amount = uint256(legs[i]) & ((1 << 96) - 1);
            require(to != address(0), "zero payee");
            require(amount > 0, "zero amount");
            total += amount;
            require(token.transferFrom(msg.sender, to, amount), "pull failed");
            emit Settled(memo, to, amount);
        }
    }
}
