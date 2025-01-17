import { Map, Set } from 'immutable'
import { SignalData } from 'simple-peer'
import { WRTC } from '@koush/wrtc'

import { Peer } from './peer'
import { NodeID } from '../types'
import { PeerConnection, EventConnection } from '../event_connection'

// TODO cleanup old peers

export class PeerPool {
  private peers = Map<NodeID, PeerConnection>()

  private constructor (
    private readonly id: NodeID,
    private readonly wrtc?: WRTC
  ) {}

  static async init (id: NodeID): Promise<PeerPool> {
    // needed on node
    let wrtc: WRTC | undefined
    try {
      // resolve relatively to where it is run, not from discojs dir
      const path = require.resolve('@koush/wrtc', { paths: ['.'] })
      wrtc = await import(path)
    } catch (e) {
      // expected
    }

    return new PeerPool(id, wrtc)
  }

  shutdown (): void {
    console.info(`[${this.id}] shutdown their peers`)

    this.peers.forEach((peer) => peer.disconnect())
    this.peers = Map()
  }

  signal (peerId: NodeID, signal: SignalData): void {
    console.info(`[${this.id}] signals for`, peerId)

    const peer = this.peers.get(peerId)
    if (peer === undefined) {
      throw new Error(`received signal for unknown peer: ${peerId}`)
    }

    peer.signal(signal)
  }

  async getPeers (
    peersToConnect: Set<NodeID>,
    signallingServer: EventConnection,
    // TODO as event?
    clientHandle: (connections: Map<NodeID, PeerConnection>) => void
  ): Promise<Map<NodeID, PeerConnection>> {
    if (peersToConnect.contains(this.id)) {
      throw new Error('peers to connect contains our id')
    }

    console.info(`[${this.id}] is connecting peers:`, peersToConnect.toJS())

    const newPeers = Map(peersToConnect
      .filter((id) => !this.peers.has(id))
      .map((id) => [id, id < this.id] as [NodeID, boolean])
      .map(([id, initiator]) => {
        const p = new Peer(id, { initiator, wrtc: this.wrtc })
        // onNewPeer(id, p)
        return [id, p]
      }))

    console.info(`[${this.id}] asked to connect new peers:`, newPeers.keySeq().toJS())
    const newPeersConnections = newPeers.map((peer, id) => new PeerConnection(this.id, peer, signallingServer))

    // adding peers to pool before connecting them because they must be set to call signal on them
    this.peers = this.peers.merge(newPeersConnections)

    clientHandle(this.peers)

    await Promise.all(
      Array.from(newPeersConnections.values()).map(async (connection) => await connection.connect()))

    console.info(`[${this.id}] knowns connected peers:`, this.peers.keySeq().toJS())

    return this.peers
      .filter((_, id) => peersToConnect.has(id))
  }
}
