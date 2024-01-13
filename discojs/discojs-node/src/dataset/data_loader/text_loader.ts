import { dataset } from '../..'
import fs from 'fs'
import {
    TextSource,
    TextConfig,
    TextLoader,
    TokenizedDataset,
} from '@epfml/discojs-core/src/dataset/data_loader/text_loader'
import { List } from 'immutable'

export class NodeTextLoader extends TextLoader {
    /**
     * Creates a file stream from a dataset filename.
     * This stream will contain a specific number of bytes
     * defined by the highWaterMark parameter which depends on the
     * block size and batch size. This ensures that reading the stream
     * always return a chunk of data of the same, required, size.
     * @param source: dataset filename to stream from
     * @param config: TextConfig
     * @returns a file stream
     */
    async getFileStream(source: string, config: TextConfig) {
        // blockSize + 1 = input size (size of x = blockSize, size of y = blockSize shifted right by 1, thus the + 1)
        // * batchSize to retrieve a batch at once
        // * 2 because tokens are stored as uint16 and thus require 2 bytes
        const highWaterMark = (config.blockSize + 1) * this.batchSize * 2 // (config.blockSize + config.batchSize + 1) * 2
        if (isNaN(highWaterMark))
            throw new Error(
                'highWaterMark, defining the stream chunk size, is NaN but is supposed to be of type number'
            )

        return new Promise<fs.ReadStream>((resolve) => {
            const stream = fs.createReadStream(source, {
                highWaterMark, // set this to seq length * 2 because we store uint16,
            })
            stream.on('readable', () => resolve(stream))
        })
    }

    /**
     * Creates an infinite iterator from a file stream
     * meaning when the stream reaches the end of the file
     * it will start again from the beginning
     * @param source: dataset filename to stream from
     * @param config: TextConfig
     * @returns an infinite iterator over a file stream
     */
    async getInfiniteBufferIteratorFromFile(
        source: string,
        config: TextConfig
    ): Promise<AsyncIterator<Buffer, Buffer, Buffer>> {
        const getStream = async () => {
            const stream = await this.getFileStream(source, config)
            return {
                stream,
                iter: stream.iterator() as AsyncIterableIterator<Buffer>,
            }
        }
        let { stream, iter } = await getStream()
        return {
            next: async () => {
                let sample = await iter.next()
                if (!sample || !sample.value || sample.done) {
                    stream.close()
                    const newStream = await getStream()
                    stream = newStream.stream
                    iter = newStream.iter
                    sample = await iter.next()
                }
                return sample
            },
        }
    }

    async load(source: string, config: TextConfig): Promise<TokenizedDataset> {
        const requestNext = await this.getInfiniteBufferIteratorFromFile(
            source,
            config
        )
        const dataset = await this.getCoreDataset(config, requestNext)
        return dataset
    }

    async loadAll(
        source: TextSource,
        config?: Partial<TextConfig>
    ): Promise<dataset.DataSplit> {
        const _config = this.resolveConfig(config)
        const split: Partial<dataset.DataSplit> = {}
        for await (const [k, files] of Object.entries(source)) {
            const datasets = await Promise.all(
                files.map(async (src) => await this.load(src, _config))
            )
            const dataset = List(datasets).reduce(
                (acc: dataset.Dataset, dataset) => acc.concatenate(dataset)
            )
            const data = await this.createData(dataset)
            ;(split as dataset.DataSplit)[k as keyof typeof split] = data
        }
        return split as dataset.DataSplit
    }
}
