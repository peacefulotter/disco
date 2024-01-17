import { Map, Set } from 'immutable'
import * as nodeUrl from 'url'

import { TrainingInformant, WeightsContainer, serialization } from '../..'
import { Client, NodeID } from '..'
import { type, ClientConnected } from '../messages'
import { timeout } from '../utils'
import {
    EventConnection,
    WebSocketServer,
    waitMessage,
    PeerConnection,
    waitMessageWithTimeout,
} from '../event_connection'
import { PeerPool } from './peer_pool'
import * as messages from './messages'

/**
 * Represents a decentralized client in a network of peers. Peers coordinate each other with the
 * help of the network's server, yet only exchange payloads between each other. Communication
 * with the server is based off regular WebSockets, whereas peer-to-peer communication uses
 * WebRTC for Node.js.
 */
export class DecentralizedClient extends Client {
    /**
     * The pool of peers to communicate with during the current training round.
     */
    private pool?: Promise<PeerPool>
    private connections?: Map<NodeID, PeerConnection>

    /**
     * Send message to server that this client is ready for the next training round.
     */
    private async waitForPeers(
        round: number
    ): Promise<Map<NodeID, PeerConnection>> {
        console.info(`[${this.ownId}] is ready for round`, round)

        // Broadcast our readiness
        const readyMessage: messages.PeerIsReady = { type: type.PeerIsReady }

        if (this.server === undefined) {
            throw new Error('server undefined, could not connect peers')
        }
        this.server.send(readyMessage)

        // Wait for peers to be connected before sending any update information
        try {
            const receivedMessage = await waitMessageWithTimeout(
                this.server,
                type.PeersForRound
            )
            if (this.nodes.size > 0) {
                throw new Error(
                    'got new peer list from server but was already received for this round'
                )
            }

            const peers = Set(receivedMessage.peers)
            console.info(
                `[${this.ownId}] received peers for round:`,
                peers.toJS()
            )
            if (this.ownId !== undefined && peers.has(this.ownId)) {
                throw new Error('received peer list contains our own id')
            }

            this.aggregator.setNodes(peers.add(this.ownId))

            if (this.pool === undefined) {
                throw new Error('waiting for peers but peer pool is undefined')
            }

            const pool = await this.pool
            const connections = await pool.getPeers(
                peers,
                this.server,
                // Init receipt of peers weights
                (conn) => this.receivePayloads(conn, round)
            )

            console.info(
                `[${this.ownId}] received peers for round ${round}:`,
                connections.keySeq().toJS()
            )
            return connections
        } catch (e) {
            console.error(e)
            this.aggregator.setNodes(Set(this.ownId))
            return Map()
        }
    }

    protected sendMessagetoPeer(
        peer: PeerConnection,
        msg: messages.PeerMessage
    ): void {
        console.info(`[${this.ownId}] send message to peer`, msg.peer, msg)
        peer.send(msg)
    }

    /**
     * Creation of the WebSocket for the server, connection of client to that WebSocket,
     * deals with message reception from the decentralized client's perspective (messages received by client).
     */
    private async connectServer(url: URL): Promise<EventConnection> {
        const server: EventConnection = await WebSocketServer.connect(
            url,
            messages.isMessageFromServer,
            messages.isMessageToServer
        )

        server.on(type.SignalForPeer, (event) => {
            console.info(`[${this.ownId}] received signal from`, event.peer)

            if (this.pool === undefined) {
                throw new Error('received signal but peer pool is undefined')
            }
            void this.pool.then((pool) => pool.signal(event.peer, event.signal))
        })

        return server
    }

    async connect(): Promise<void> {
        const URL = typeof window !== 'undefined' ? window.URL : nodeUrl.URL
        const serverURL = new URL('', this.url.href)
        switch (this.url.protocol) {
            case 'http:':
                serverURL.protocol = 'ws:'
                break
            case 'https:':
                serverURL.protocol = 'wss:'
                break
            default:
                throw new Error(`unknown protocol: ${this.url.protocol}`)
        }
        serverURL.pathname += `deai/${this.task.id}`

        this._server = await this.connectServer(serverURL)

        const msg: ClientConnected = {
            type: type.ClientConnected,
        }
        this.server.send(msg)

        const peerIdMsg = await waitMessage(this.server, type.AssignNodeID)
        console.info(`[${peerIdMsg.id}] assigned id generated by server`)

        if (this._ownId !== undefined) {
            throw new Error('received id from server but was already received')
        }
        this._ownId = peerIdMsg.id
        this.pool = PeerPool.init(peerIdMsg.id)
    }

    async disconnect(): Promise<void> {
        // Disconnect from peers
        const pool = await this.pool
        pool?.shutdown()
        this.pool = undefined

        if (this.connections !== undefined) {
            const peers = this.connections.keySeq().toSet()
            this.aggregator.setNodes(this.aggregator.nodes.subtract(peers))
        }

        // Disconnect from server
        this.server?.disconnect()
        this._server = undefined
        this._ownId = undefined
    }

    async onRoundBeginCommunication(
        weights: WeightsContainer,
        round: number,
        trainingInformant: TrainingInformant
    ): Promise<void> {
        // Reset peers list at each round of training to make sure client works with an updated peers
        // list, maintained by the server. Adds any received weights to the aggregator.
        this.connections = await this.waitForPeers(round)
        // Store the promise for the current round's aggregation result.
        this.aggregationResult = this.aggregator.receiveResult()
    }

    async onRoundEndCommunication(
        weights: WeightsContainer,
        round: number,
        trainingInformant: TrainingInformant
    ): Promise<void> {
        let result = weights

        // Perform the required communication rounds. Each communication round consists in sending our local payload,
        // followed by an aggregation step triggered by the receipt of other payloads, and handled by the aggregator.
        // A communication round's payload is the aggregation result of the previous communication round. The first
        // communication round simply sends our training result, i.e. model weights updates. This scheme allows for
        // the aggregator to define any complex multi-round aggregation mechanism.
        for (let r = 0; r < this.aggregator.communicationRounds; r++) {
            // Generate our payloads for this communication round and send them to all ready connected peers
            if (this.connections !== undefined) {
                const payloads = this.aggregator.makePayloads(result)
                try {
                    await Promise.all(
                        payloads.map(async (payload, id) => {
                            if (id === this.ownId) {
                                this.aggregator.add(
                                    this.ownId,
                                    payload,
                                    round,
                                    r
                                )
                            } else {
                                const connection = this.connections?.get(id)
                                if (connection !== undefined) {
                                    const encoded =
                                        await serialization.weights.encode(
                                            payload
                                        )
                                    this.sendMessagetoPeer(connection, {
                                        type: type.Payload,
                                        peer: id,
                                        round: r,
                                        payload: encoded,
                                    })
                                }
                            }
                        })
                    )
                } catch {
                    throw new Error('error while sending weights')
                }
            }

            if (this.aggregationResult === undefined) {
                throw new TypeError('aggregation result promise is undefined')
            }

            // Wait for aggregation before proceeding to the next communication round.
            // The current result will be used as payload for the eventual next communication round.
            result = await Promise.race([this.aggregationResult, timeout()])

            // There is at least one communication round remaining
            if (r < this.aggregator.communicationRounds - 1) {
                // Reuse the aggregation result
                this.aggregationResult = this.aggregator.receiveResult()
            }
        }

        // Reset the peers list for the next round
        this.aggregator.resetNodes()
    }

    private receivePayloads(
        connections: Map<NodeID, PeerConnection>,
        round: number
    ): void {
        console.info(
            `[${this.ownId}] Accepting new contributions for round ${round}`
        )
        connections.forEach(async (connection, peerId) => {
            let receivedPayloads = 0
            do {
                try {
                    const message = await waitMessageWithTimeout(
                        connection,
                        type.Payload
                    )
                    const decoded = serialization.weights.decode(
                        message.payload
                    )

                    if (
                        !this.aggregator.add(
                            peerId,
                            decoded,
                            round,
                            message.round
                        )
                    ) {
                        console.warn(
                            `[${this.ownId}] Failed to add contribution from peer ${peerId}`
                        )
                    }
                } catch (e) {
                    console.warn(e instanceof Error ? e.message : e)
                }
            } while (++receivedPayloads < this.aggregator.communicationRounds)
        })
    }
}
