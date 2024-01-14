import { WebSocket, type MessageEvent } from 'ws'
import { v4 as randomUUID } from 'uuid'
import { dataset } from '../..'

class Deferred<T> {
    promise: Promise<T>
    resolve: (value: T | PromiseLike<T>) => void = () => {}
    reject: (reason?: any) => void = () => {}

    constructor() {
        this.promise = new Promise<T>((resolve, reject) => {
            this.resolve = resolve
            this.reject = reject
        })
    }
}

class Cache<E> {
    position: number = 0
    private readonly cache: (Deferred<E> | E)[]

    private constructor(
        private readonly length: number,
        private readonly request: (pos: number) => void | Promise<void>
    ) {
        this.cache = Array.from({ length }, () => new Deferred<E>())
    }

    // pre-loads the cache with the first n requests
    // To my knowledge, this is the only way to do this
    // since you can't have parallelism in the websocket server
    // otherwise you end up with multiple times the same sample
    // in the cache and some promises are never resolved
    static async init<E>(
        length: number,
        request: (pos: number) => void | Promise<void>,
        initializer: (c: Cache<E>) => void
    ): Promise<Cache<E>> {
        const cache = new Cache<E>(length, request)
        initializer(cache)
        for (let pos = 0; pos < length; pos++) {
            cache.request(pos)
        }
        return cache
    }

    put(pos: number, elt: E): void {
        const promise = this.cache[pos] as Deferred<E>
        promise.resolve(elt)
    }

    async next(): Promise<E> {
        const eltOrDeffered = this.cache[this.position]
        const elt =
            eltOrDeffered instanceof Deferred
                ? await eltOrDeffered.promise
                : eltOrDeffered

        const pos = this.position
        this.cache[pos] = new Deferred<E>()
        this.request(pos)
        this.position = (pos + 1) % this.length
        return elt
    }
}

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
    static readonly CACHE_SIZE: number = 10
    websockets: WebSocket[] = []

    /**
     * Builds a URL with the following search parameters
     * @param file: filename corresponding to the file the websocket server will stream
     * @param config entries: all the config key, value pairs. The config object will be reconstructed in the websocket server side
     */
    getWebSocket = (file: string, config: dataset.TextConfig) =>
        new Promise<{ ws: WebSocket; id: string }>((resolve) => {
            const url = new URL(WebTextLoader.BROKER_URL)

            const id = randomUUID()
            const searchParams: dataset.WSSearchParams = {
                id,
                config: JSON.stringify(config),
                file,
            }
            for (const [k, v] of Object.entries(searchParams))
                url.searchParams.append(k, v)

            const ws = new WebSocket(url)
            ws.onopen = () => {
                resolve({ ws, id })
            }
        })

    async load(
        file: string,
        config: dataset.TextConfig
    ): Promise<dataset.TokenizedDataset> {
        // TODO: /!\ implement a way to close websocket at the end of training
        // onTrainEnd = () => ws.close()
        const { ws, id } = await this.getWebSocket(file, config)

        const cache = await Cache.init<IteratorResult<number[], number[]>>(
            WebTextLoader.CACHE_SIZE,
            (pos) => ws.send(JSON.stringify({ pos, id })),
            (c) => {
                ws.onmessage = (payload: MessageEvent) => {
                    const { value, done, pos } = JSON.parse(
                        payload.data as string
                    ) as MessageData
                    c.put(pos, { value: value.data, done })
                }
            }
        )

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
