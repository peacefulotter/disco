import { dataset } from '../..'
import fs from 'node:fs'
import {
    TextSource,
    TextConfig,
    TextLoader,
    TokenizedDataset,
} from '@epfml/discojs-core/src/dataset/data_loader/text_loader'
import { List } from 'immutable'

export class NodeTextLoader extends TextLoader {
    getFileStream(source: string, config: TextConfig) {
        // blockSize + 1 = input size (size of x = blockSize, size of y = blockSize shifted right by 1, thus the + 1)
        // * batchSize to retrieve a batch at once
        // * 2 because tokens are stored as uint16 and thus require 2 bytes
        const highWaterMark = (config.blockSize + 1) * this.batchSize * 2 // (config.blockSize + config.batchSize + 1) * 2
        return fs.createReadStream(source, {
            highWaterMark, // set this to seq length * 2 because we store uint16,
        })
    }

    getIteratorDatasetFromFile(source: string, config: TextConfig): AsyncIterator<Buffer> {
        const getStream = () => {
            const stream = this.getFileStream(source, config)
            return {
                stream,
                iter: stream.iterator() as AsyncIterableIterator<Buffer>,
            }
        }
        let { stream, iter } = getStream()
        return {
            next: async () => {
                let sample = await iter.next()
                if (sample.done) {
                    stream.close()
                    const newStream = getStream()
                    stream = newStream.stream
                    iter = newStream.iter
                    sample = await iter.next()
                }
                return sample
            },
        }
    }

    async loadDatasetFrom(source: string, config: TextConfig): Promise<TokenizedDataset> {
        const prefix = 'file://'
        if (!source.startsWith(prefix)) {
            source = prefix + source
        }

        let stream = this.getIteratorDatasetFromFile(source, config)
        const requestNext = async () => {
            const { value } = await stream.next()
            return value
        }
        const dataset = await this.getCoreDataset(config, requestNext)
        return dataset
    }

    async load(source: string, config: TextConfig): Promise<TokenizedDataset> {
        const ds = await this.loadDatasetFrom(source, config)
        return ds
    }

    async loadAll(source: TextSource, config?: Partial<TextConfig>): Promise<dataset.DataSplit> {
        const _config = this.resolveConfig(config)
        const split: Partial<dataset.DataSplit> = {}
        for await (const [k, files] of Object.entries(source)) {
            const datasets = await Promise.all(
                files.map(async (src) => await this.load(src, _config))
            )
            const dataset = List(datasets).reduce((acc: dataset.Dataset, dataset) =>
                acc.concatenate(dataset)
            )
            const data = await this.createData(dataset)
            ;(split as dataset.DataSplit)[k as keyof typeof split] = data
        }
        return split as dataset.DataSplit
    }
}
