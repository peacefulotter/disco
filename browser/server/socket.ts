import path from 'path'
import { startDisco } from '@epfml/disco-server'
import { dataset, defaultTasks, node } from '@epfml/discojs-node'

const getParams = (searchParams: URLSearchParams) => {
    const obj = Object.fromEntries(searchParams) as dataset.WSSearchParams
    const params: dataset.ParsedWSSearchParams = {
        id: obj.id,
        config: JSON.parse(obj.config) as dataset.TextConfig,
        file: path.join(import.meta.dir, obj.file), // since we don't have access to the working dir in the browser
    }
    return params
}

type WebsocketStatus = {
    iterator: AsyncIterator<Buffer, Buffer, Buffer>
    next: Promise<IteratorResult<Buffer, Buffer>>
}

const database: Record<string, WebsocketStatus> = {}

// TODO: make this generic and dependent on url
// A mapping between taskId and defaultTask would be enough
// what about custom tasks though?
const task = defaultTasks.wikitext.getTask()

Bun.serve({
    async fetch(req, server) {
        const url = new URL(req.url)
        const { id, config, file } = getParams(url.searchParams)
        const loader = new node.dataset.loader.NodeTextLoader(task)
        const iterator = await loader.getInfiniteBufferIteratorFromFile(
            file,
            config
        )
        const next = iterator.next()
        database[id] = { iterator, next }
        server.upgrade(req)
    },
    websocket: {
        async message(ws, payload) {
            const { id, pos } = JSON.parse(payload as string) as {
                pos: number
                id: string
            }
            const status = database[id]
            const data = await status.next
            ws.send(JSON.stringify({ ...data, pos }))

            // same as in core text-loader, we pre-fetch the next chunk even before actually requesting it
            status.next = status.iterator.next()
        },
        async open(ws) {
            console.log('[WebSocketServer] Connection with client established')
        },
        async close(ws) {
            // TODO: on websocket close, remove from database
            console.log('[WebSocketServer] Bye')
        },
    },
    // TODO: store this and URL in .env to share URL between web/ and server/
    port: process.env.PORT || 3001,
})

await startDisco()
