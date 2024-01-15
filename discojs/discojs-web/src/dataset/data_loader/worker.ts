// prevents TS errors
declare var self: Worker

import { v4 as randomUUID } from 'uuid'
import { dataset } from '../..'
import { Cache, Deferred } from './cache'

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

    const request = (pos: number) => {
        // console.log(Date.now(), 'WORKER requesting next value', pos)
        ws.send(JSON.stringify({ pos, id }))
    }

    const cache = await Cache.init<CacheData>(
        parseInt(CACHE_SIZE),
        request,
        () => {},
        0
    )

    ws.onmessage = (payload: MessageEvent) => {
        // console.log(
        //     Date.now(),
        //     'WORKER received from ws',
        //     JSON.parse(payload.data).pos
        // )
        postMessage(payload.data)
    }

    self.onmessage = async (event: MessageEvent<string>) => {
        // console.log(Date.now(), 'WORKER onmessage')
        request(cache.position)
        cache.position = (cache.position + 1) % cache.length
    }

    // self.postMessage('done')

    // ================================================

    // const cache = new Deferred<any>()

    // ws.onmessage = (payload: MessageEvent) => {
    //     const { value, done, pos } = JSON.parse(
    //         payload.data as string
    //     ) as MessageData
    //     console.log(Date.now(), 'WORKER got value for', pos)
    //     cache.resolve({ value: value.data, done, pos })
    // }

    // const next = async (pos: number) => {
    //     const element = await cache.promise
    //     cache.reset()
    //     console.log(Date.now(), 'WORKER asked', pos, ' returns', element.pos)
    //     ws.send(JSON.stringify({ pos, id }))
    //     return element
    // }

    // self.onmessage = async (event: MessageEvent<string>) => {
    //     const pos = parseInt(event.data)
    //     console.log(Date.now(), 'WORKER requesting', pos)
    //     const payload = await next(pos)
    //     // console.log(Date.now(), 'WORKER onmessage ', payload.pos)
    //     postMessage(JSON.stringify(payload))
    // }

    // ws.send(JSON.stringify({ pos: -1, id }))
    // self.postMessage('done')
}

ws.onopen = () => {
    self.onmessage = async (event: MessageEvent<string>) => {
        // receiving init
        proceed()
    }
    self.postMessage('connected')
}
