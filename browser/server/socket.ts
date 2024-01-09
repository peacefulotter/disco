import { WebSocketServer } from 'ws'
import { Server } from './types.js'
import { defaultTasks, dataset, node } from '@epfml/discojs-node'
import { IncomingMessage } from 'http'

// TODO: store this in .env to share URL between web/ and server/
const baseUrl = 'http://localhost:3001'

const getParams = (req: IncomingMessage) => {
    const { searchParams } = new URL(`${baseUrl}${req.url}`)
    const obj = Object.fromEntries(searchParams)
    const params = {
        ...obj,
        blockSize: parseInt(obj.blockSize),
        vocabSize: parseInt(obj.vocabSize),
    } as dataset.WSSearchParams
    console.log('WS Received params:', params)
    return params
}

const initWebsockets = async (server: Server) => {
    const wss = new WebSocketServer({ server })

    wss.on('connection', async (ws, req) => {
        console.log('Connection with client established')

        const { taskId, file, ...config } = getParams(req)
        const task = defaultTasks[taskId].getTask()
        const loader = new node.dataset.loader.NodeTextLoader(task)
        const dataset = await loader.load(file, config)
        const iterator = await dataset.iterator()

        ws.addEventListener('message', async (event) => {
            const { value } = await iterator.next()
            ws.send(JSON.stringify(value))
        })
    })
}

export default initWebsockets
