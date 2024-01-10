import 'module-alias/register'
import express from 'express'
import * as http from 'http'
import initWebsockets from './socket'
import { startDisco } from './disco'

// TODO: store this and URL in .env to share URL between web/ and server/
const PORT = process.env.PORT || 3001

const app = express()

app.use(
    express.urlencoded({
        extended: true,
    })
)

app.use(
    express.json({
        type: ['application/json', 'text/plain'],
        limit: '200mb',
    })
)

const server = http.createServer(app)

// start our server
server.listen(PORT, async () => {
    console.log(`Server listening on ${PORT}`)
    initWebsockets(server)
    await startDisco()
})
