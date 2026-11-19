// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

/// Native-currency payment streams, each paid out linearly to its recipient between a start and a stop time.
contract PaymentStreams {
    struct Stream {
        address sender;
        address recipient;
        uint128 deposit;
        uint128 withdrawn;
        uint64 start;
        uint64 stop;
        bool cancelled;
    }

    Stream[] private _streams;
    bool private _inited;
    uint32 public minDuration;

    event StreamCreated(
        uint256 indexed id, address indexed sender, address indexed recipient, uint256 deposit, uint64 start, uint64 stop
    );
    event Withdrawn(uint256 indexed id, address indexed recipient, uint256 amount);
    event Cancelled(uint256 indexed id, uint256 toRecipient, uint256 refund);

    function initialize(uint32 minDuration_) external {
        require(!_inited, "already initialized");
        _inited = true;
        minDuration = minDuration_;
    }

    /// Opens a stream of msg.value from the caller to `recipient`, paid out linearly between `start` and `stop`.
    function createStream(address recipient, uint64 start, uint64 stop) external payable returns (uint256 id) {
        require(recipient != address(0) && recipient != address(this) && recipient != msg.sender, "bad recipient");
        require(stop > start && stop - start >= minDuration, "duration too short");
        require(start + 1 days >= block.timestamp && start <= block.timestamp + 30 days, "start out of range");
        require(msg.value > 0 && msg.value % (stop - start) == 0, "deposit must be a multiple of the duration in seconds");
        id = _streams.length;
        _streams.push(Stream(msg.sender, recipient, uint128(msg.value), 0, start, stop, false));
        emit StreamCreated(id, msg.sender, recipient, msg.value, start, stop);
    }

    function streamCount() external view returns (uint256) {
        return _streams.length;
    }

    function getStream(uint256 id) external view returns (Stream memory) {
        return _streams[id];
    }

    function ratePerSecond(uint256 id) public view returns (uint256) {
        Stream storage s = _streams[id];
        return s.deposit / (s.stop - s.start);
    }

    function streamed(uint256 id) public view returns (uint256) {
        Stream storage s = _streams[id];
        if (s.cancelled || block.timestamp <= s.start) return s.cancelled ? s.withdrawn : 0;
        uint256 end = block.timestamp < s.stop ? block.timestamp : s.stop;
        return (end - s.start) * ratePerSecond(id);
    }

    function withdrawable(uint256 id) public view returns (uint256) {
        return streamed(id) - _streams[id].withdrawn;
    }

    function withdraw(uint256 id, uint256 amount) external {
        Stream storage s = _streams[id];
        require(msg.sender == s.recipient, "only the recipient");
        require(amount <= withdrawable(id), "more than withdrawable");
        s.withdrawn += uint128(amount);
        _pay(s.recipient, amount);
        emit Withdrawn(id, s.recipient, amount);
    }

    /// Stops a stream: what has streamed so far goes to the recipient, the rest back to the sender.
    function cancel(uint256 id) external {
        Stream storage s = _streams[id];
        require(msg.sender == s.sender, "only the sender");
        require(!s.cancelled, "already cancelled");
        uint256 owed = withdrawable(id);
        uint256 refund = s.deposit - s.withdrawn - owed;
        s.withdrawn += uint128(owed);
        s.cancelled = true;
        if (owed > 0) _pay(s.recipient, owed);
        if (refund > 0) _pay(s.sender, refund);
        emit Cancelled(id, owed, refund);
    }

    function _pay(address to, uint256 amount) private {
        (bool ok,) = to.call{value: amount}("");
        require(ok, "payment failed");
    }
}
