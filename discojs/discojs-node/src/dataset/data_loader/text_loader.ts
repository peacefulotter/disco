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
        if (isNaN(highWaterMark))
            throw new Error(
                'highWaterMark, defining the stream chunk size, is NaN but is supposed to be of type number'
            )
        return fs.createReadStream(source, {
            highWaterMark, // set this to seq length * 2 because we store uint16,
        })
    }

    getInfiniteIteratorFromFile(
        source: string,
        config: TextConfig
    ): AsyncIterator<Buffer, Buffer, Buffer> {
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

    async getFileStreamIterator(source: string, config: TextConfig) {
        const prefix = 'file://'
        if (!source.startsWith(prefix)) {
            source = prefix + source
        }

        let stream = this.getInfiniteIteratorFromFile(source, config)

        const requestNext = async (): Promise<number[]> => {
            const { value } = await stream.next()
            const chunk = value.toJSON().data
            return chunk
        }
        return requestNext
    }

    async load(source: string, config: TextConfig): Promise<TokenizedDataset> {
        const requestNext = await this.getFileStreamIterator(source, config)
        const dataset = await this.getCoreDataset(config, requestNext)
        return dataset
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
