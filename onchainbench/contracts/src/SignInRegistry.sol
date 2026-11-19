// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.26;

/// An on-chain sign-in registry: a wallet logs in by submitting a challenge from its own address.
contract SignInRegistry {
    string public site;
    mapping(address => bytes32) public sessionOf;
    mapping(address => uint256) public loggedInAt;

    event LoggedIn(address indexed wallet, bytes32 challenge);

    constructor(string memory s) {
        site = s;
    }

    function login(bytes32 challenge) external {
        require(challenge != bytes32(0), "empty challenge");
        sessionOf[msg.sender] = challenge;
        loggedInAt[msg.sender] = block.timestamp;
        emit LoggedIn(msg.sender, challenge);
    }
}
