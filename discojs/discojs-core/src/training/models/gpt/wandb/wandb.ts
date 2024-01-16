import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'

import { GPTConfig } from '@/model'

export type WandbConfig = GPTConfig & {
    platform: string
    backend: string
    gpu: string
    model: string
}

export type WandbSave = {
    init: {
        config: WandbConfig
        date: string
    }
    logs: any[]
}

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
        `disco_${save.init.config.platform}_${save.init.config.backend}_${save.init.config.gpu}_${save.init.config.model}.json`
    )
    await fs.writeFile(p, json, 'utf-8')
}

export class Wandb {
    public save: WandbSave
    public config: WandbConfig

    constructor(config: WandbConfig) {
        const date = new Date().toISOString()
        this.save = {
            init: {
                config,
                date,
            },
            logs: [],
        }
        this.config = config
    }

    public async log(payload: any) {
        this.save.logs.push(payload)
    }

    public async finish() {
        // POST request the Disco server on path 'wandb'
        // This requires to attach a post request for this corresponding path to the Disco
        // server (see ~/server/src/start_server.ts)
        // TODO: store url in .env file or find a different way to make this automatically compatible with Disco server
        console.log(this.save)
        await exportWandb(this.save)
        await fetch('http://localhost:8000/wandb', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                save: this.save,
            }),
        })
    }
}
