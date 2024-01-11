export type WandbConfig = { platform: string; gpu: string; model: string }

export type WandbSave = {
    init: {
        config: WandbConfig
        date: string
    }
    logs: any[]
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
