// prevents TS errors
declare var self: Worker

import { v4 as randomUUID } from 'uuid'
import { dataset } from '../..'
import { Cache } from './cache'

type MessageData = {
    value: {
        type: 'Buffer'
        data: number[]
    }
    done: boolean
    pos: number
}

export type CacheData = {
    value: number[]
    done: boolean
    pos: number
}

const BROKER_URL = 'ws://localhost:3001/ws'

const { FILE, CONFIG, CACHE_SIZE } = process.env as {
    ID: string
    FILE: string
    CONFIG: string
    CACHE_SIZE: string
}

const url = new URL(BROKER_URL)

const id = randomUUID()
const searchParams: dataset.WSSearchParams = {
    id,
    config: CONFIG,
    file: FILE,
}
for (const [k, v] of Object.entries(searchParams)) url.searchParams.append(k, v)

const ws = new WebSocket(url)

ws.onerror = (err) => {
    console.error(err)
}

const proceed = async () => {
    console.log('worker', id, 'connected')

    const cache = await Cache.init<CacheData>(
        parseInt(CACHE_SIZE),
        (pos) => ws.send(JSON.stringify({ pos, id })),
        (c) => {
            ws.onmessage = (payload: MessageEvent) => {
                const { value, done, pos } = JSON.parse(
                    payload.data as string
                ) as MessageData
                // console.log(Date.now(), 'WORKER got value for', pos)
                c.put(pos, { value: value.data, done, pos })
            }
        }
    )

    self.onmessage = async (event: MessageEvent<string>) => {
        const payload = await cache.next()
        // console.log(Date.now(), 'WORKER onmessage ', payload.pos)
        postMessage(JSON.stringify(payload))
    }

    self.postMessage('done')
}

ws.onopen = proceed
