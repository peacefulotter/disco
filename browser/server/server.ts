import 'module-alias/register'
import express from 'express'
import * as http from 'http'
import initWebsockets from './socket'

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
server.listen(PORT, () => {
    console.log(`Server listening on ${PORT}`)
    initWebsockets(server)
})
