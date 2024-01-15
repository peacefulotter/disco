import { dataset } from '../..'
import { Cache } from './cache'
import { CacheData } from './worker'

export class WebTextLoader extends dataset.loader.TextLoader {
    static readonly CACHE_SIZE: number = 10

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

        const worker = await this.getWorker(file, config)

        const cache = await Cache.init<CacheData>(
            WebTextLoader.CACHE_SIZE,
            (pos, init) => {
                worker.postMessage(JSON.stringify({ pos, init }))
            },
            (c) => {
                worker.onmessage = (payload: globalThis.MessageEvent<any>) => {
                    const sample = JSON.parse(
                        payload.data as string
                    ) as CacheData
                    c.put(sample.pos, sample)
                }
            }
        )

        // iterator just to have a way to console.time the next() call
        const iterator = {
            next: async () => {
                // console.time('wait')
                const sample = await cache.next()
                //  console.timeEnd('wait')
                return sample
            },
        }

        const dataset = await this.getCoreDataset(config, iterator)
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
