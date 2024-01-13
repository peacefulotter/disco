import { WebSocket, type MessageEvent } from 'ws'
import { dataset } from '../..'

export class WebTextLoader extends dataset.loader.TextLoader {
    websockets: WebSocket[] = []

    /**
     * Builds a URL with the following search parameters
     * taskId: The task ID as defined by this.task.taskID
     * file: filename corresponding to the file the websocket server will stream
     * config entries: all the config key, value pairs. The config object will be reconstructed in the websocket server side
     */
    getWebSocket = async (file: string, config: dataset.TextConfig) =>
        new Promise<WebSocket>((resolve) => {
            const brokerURL = new URL('ws://localhost:3001/ws')
            const searchParams: dataset.WSSearchParams = {
                task: JSON.stringify(this.task),
                config: JSON.stringify(config),
                file,
            }
            for (const [k, v] of Object.entries(searchParams))
                brokerURL.searchParams.append(k, v)
            const ws = new WebSocket(brokerURL)
            ws.onopen = () => {
                resolve(ws)
            }
        })

    async load(
        file: string,
        config: dataset.TextConfig
    ): Promise<dataset.TokenizedDataset> {
        // TODO: /!\ implement a way to close websocket at the end of training
        // onTrainEnd = () => ws.close()
        const ws = await this.getWebSocket(file, config)

        const iterator = {
            next: () =>
                new Promise<IteratorResult<Buffer, Buffer>>((resolve) => {
                    // console.time('ws send and receive')
                    ws.onmessage = (payload: MessageEvent) => {
                        // const data = JSON.parse(payload.data as string)
                        // console.timeEnd('ws send and receive')
                        resolve({ value: payload.data as Buffer, done: false })
                    }
                    ws.send('req')
                }),
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
