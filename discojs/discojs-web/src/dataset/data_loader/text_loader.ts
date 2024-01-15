// import { WebSocket } from 'ws'
// import { v4 as randomUUID } from 'uuid'
import { dataset } from '../..'
import { Cache } from './cache'
import { CacheData } from './worker'

type MessageData = {
    value: {
        type: 'Buffer'
        data: number[]
    }
    done: boolean
    pos: number
}

export class WebTextLoader extends dataset.loader.TextLoader {
    // TODO: make brokerURL configurable and at least stored in .env
    // or automatically retrieved and compatible with websocket server somehow
    static readonly BROKER_URL = 'ws://localhost:3001/ws'
    static readonly CACHE_SIZE: number = 100

    // /**
    //  * Builds a URL with the following search parameters
    //  * @param file: filename corresponding to the file the websocket server will stream
    //  * @param config entries: all the config key, value pairs. The config object will be reconstructed in the websocket server side
    //  */
    // getWebSocket = (file: string, config: dataset.TextConfig) =>
    //     new Promise<{ ws: WebSocket; id: string }>((resolve) => {
    //         const url = new URL(WebTextLoader.BROKER_URL)

    //         const id = randomUUID()
    //         const searchParams: dataset.WSSearchParams = {
    //             id,
    //             config: JSON.stringify(config),
    //             file,
    //         }
    //         for (const [k, v] of Object.entries(searchParams))
    //             url.searchParams.append(k, v)

    //         const ws = new WebSocket(url)
    //         ws.onopen = () => {
    //             resolve({ ws, id })
    //         }
    //     })

    getWorker = (file: string, config: dataset.TextConfig) => {
        const workerURL = new URL('worker.ts', import.meta.url).href
        const worker = new Worker(workerURL, {
            env: {
                FILE: file,
                CONFIG: JSON.stringify(config),
                CACHE_SIZE: WebTextLoader.CACHE_SIZE,
            },
        } as WorkerOptions)

        return new Promise<Worker>((resolve) => {
            // waiting for a message from the worker to inform the loader
            // that the websocket connection is opened
            worker.onmessage = () => {
                resolve(worker)
            }
        })
    }

    async load(
        file: string,
        config: dataset.TextConfig
    ): Promise<dataset.TokenizedDataset> {
        // TODO: /!\ implement a way to close websocket at the end of training
        // onTrainEnd = () => ws.close()

        // const { ws, id } = await this.getWebSocket(file, config)

        const worker = await this.getWorker(file, config)

        const cache = await Cache.init<IteratorResult<number[], number[]>>(
            WebTextLoader.CACHE_SIZE,
            (pos, init) => {
                if (!init) worker.postMessage(JSON.stringify({ pos, init }))
                // console.log(Date.now(), 'WS requesting next value')
            },
            (c) => {},
            1
        )

        worker.onmessage = (payload: globalThis.MessageEvent<any>) => {
            const { value, done, pos } = JSON.parse(
                payload.data as string
            ) as MessageData
            // console.log('LOADER got value for', pos)
            cache.put(pos, { value: value.data, done })
        }

        // Inform the worker that he can now start to send messages
        worker.postMessage('init')

        // const cache = await Cache.init<IteratorResult<number[], number[]>>(
        //     WebTextLoader.CACHE_SIZE,
        //     (pos) => ws.send(JSON.stringify({ pos, id })),
        //     (c) => {
        //         ws.onmessage = (payload: MessageEvent) => {
        //             const { value, done, pos } = JSON.parse(
        //                 payload.data as string
        //             ) as MessageData
        //             // console.log(Date.now(), 'got value for', pos)
        //             c.put(pos, { value: value.data, done })
        //         }
        //     }
        // )

        const dataset = await this.getCoreDataset(config, cache)
        return dataset
    }

    async loadAll(
        source: dataset.TextSource,
        config?: Partial<dataset.TextConfig> | undefined
    ): Promise<dataset.DataSplit> {
        const _config = this.resolveConfig(config)

        const loadFromSources = async (files: string[]) => {
            const datasets = await Promise.all(
                files.map((f) => this.load(f, _config)) ?? []
            )
            const ds =
                datasets.length > 1
                    ? datasets
                          .slice(1)
                          .reduce(
                              (acc, cur) => acc.concatenate(cur),
                              datasets[0]
                          )
                    : datasets[0]
            return await dataset.TextData.init(ds, this.task)
        }

        return {
            train: await loadFromSources(source.train),
            validation:
                source.validation && (await loadFromSources(source.validation)),
        }
    }
}
