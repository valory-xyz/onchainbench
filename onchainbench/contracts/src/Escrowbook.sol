// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

interface IERC20 {
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
}

/// A token escrow: scheduled transfers, each released once its unlock time has passed.
contract Escrowbook {
    struct Item {
        address to;
        uint96 amount;
        uint32 unlockAt;
    }

    IERC20 public immutable token;
    Item[] private _items;
    mapping(uint256 => bool) public paid;

    event Scheduled(uint256 indexed id, address indexed to, uint96 amount, uint32 unlockAt);
    event Released(uint256 indexed id, address indexed to, uint96 amount);

    constructor(IERC20 t) {
        token = t;
    }

    /// Pulls the sum of all amounts from the caller and records every item. Returns the id of the first.
    function scheduleBatch(Item[] calldata batch) external returns (uint256 firstId) {
        firstId = _items.length;
        uint256 total;
        for (uint256 i = 0; i < batch.length; i++) {
            total += batch[i].amount;
            _items.push(batch[i]);
            emit Scheduled(firstId + i, batch[i].to, batch[i].amount, batch[i].unlockAt);
        }
        require(token.transferFrom(msg.sender, address(this), total), "pull failed");
    }

    function releaseItem(uint256 id) external {
        Item memory it = _items[id];
        require(!paid[id], "already paid");
        require(block.timestamp >= it.unlockAt, "locked");
        paid[id] = true;
        require(token.transfer(it.to, it.amount), "push failed");
        emit Released(id, it.to, it.amount);
    }

    function itemCount() external view returns (uint256) {
        return _items.length;
    }

    function itemAt(uint256 id) external view returns (Item memory) {
        return _items[id];
    }
}
