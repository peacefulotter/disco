import path from 'path'
import { WebSocketServer } from 'ws'
import { Server } from './types.js'
import { dataset, node, Task } from '@epfml/discojs-node'
import { IncomingMessage } from 'http'

const dummyUrl = 'http://localhost:3001' // any valid url, it DOES NOT matter

const getParams = (req: IncomingMessage) => {
    const { searchParams } = new URL(`${dummyUrl}${req.url}`)
    const obj = Object.fromEntries(searchParams) as dataset.WSSearchParams
    const params = {
        task: JSON.parse(obj.task) as Task,
        config: JSON.parse(obj.config) as dataset.TextConfig,
        file: path.join(import.meta.dir, obj.file), // since we don't have access to the working dir in the browser
    } as dataset.ParsedWSSearchParams
    return params
}

const initWebsockets = async (server: Server) => {
    const wss = new WebSocketServer({ server })

    wss.on('connection', async (ws, req) => {
        console.log('Connection with client established')

        const { task, config, file } = getParams(req)
        const loader = new node.dataset.loader.NodeTextLoader(task)
        const requestNext = await loader.getFileStreamIterator(file, config)

        ws.addEventListener('message', async () => {
            const chunk = await requestNext()
            ws.send(JSON.stringify(chunk))
        })
    })
}

export default initWebsockets
