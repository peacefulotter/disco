import http from 'node:http'
import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'

import { Disco } from '.'

const exportWandb = async (save: any) => {
    const json = JSON.stringify(save, null, 4)

    const __filename = fileURLToPath(import.meta.url)
    const dir = path.join(
        path.dirname(__filename),
        '..',
        '..',
        'discojs',
        'gpt-tfjs',
        'wandb'
    )
    await fs.mkdir(dir, { recursive: true }).catch(console.error)

    const p = path.join(
        dir,
        `disco_${save.init.config.platform}_${save.init.config.gpu}_${save.init.config.model}.json`
    )
    await fs.writeFile(p, json, 'utf-8')
}

export async function startDisco(): Promise<[http.Server, URL]> {
    const disco = new Disco()
    await disco.addDefaultTasks()

    const server = disco.serve(8000)

    // Attach a POST request handler to Disco server
    // This allows to save the WandB data collected during training to a file
    disco.server.post('/wandb', async (req, res) => {
        const { save } = req.body
        await exportWandb(save)
        res.send('ok')
    })

    await new Promise((resolve, reject) => {
        server.once('listening', resolve)
        server.once('error', reject)
        server.on('error', console.error)
    })

    let addr: string
    const rawAddr = server.address()

    if (rawAddr === null) {
        throw new Error('unable to get server address')
    } else if (typeof rawAddr === 'string') {
        addr = rawAddr
    } else if (typeof rawAddr === 'object') {
        if (rawAddr.family === '4') {
            addr = `${rawAddr.address}:${rawAddr.port}`
        } else {
            addr = `[${rawAddr.address}]:${rawAddr.port}`
        }
    } else {
        throw new Error('unable to get address to server')
    }

    // return [server, new URL('', `http://${addr}`)]
    return [server, new URL('', 'http://localhost:8000')]
}
