import { tf } from '../..'
import fs from 'node:fs'
import { TextConfig, TextLoader } from '@epfml/discojs-core/src/dataset/data_loader/text_loader'
import { Dataset } from '@epfml/discojs-core/src/dataset/'

type TokenizedSample = {
    xs: number[]
    ys: number[]
}

type AsyncTokenizedGenerator = AsyncGenerator<TokenizedSample, void, unknown>

export class NodeTextLoader extends TextLoader {
    getFileStream(source: string, config: TextConfig) {
        // blockSize to get an initial full x
        // + batchSize to retrieve a batch from it (each element of the batch is a shift by << n for the nth elt in batch)
        // + 1 for the last y of the batch
        // * 2 because we store the tokens into 2 bytes (16 bites uint)
        // TODO: sure about this?
        const highWaterMark = (config.blockSize + 1) * config.batchSize * 2 // (config.blockSize + config.batchSize + 1) * 2
        console.log('highWaterMark', highWaterMark)

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

    async getBackboneDataset(config: TextConfig, requestNext: () => Promise<number[]>) {
        const { vocabSize } = config
        const sampleSize = config.blockSize + 1

        const toUInt16 = (low: number, high: number) => {
            low &= 0xff
            high &= 0xff
            return (high << 8) | low
        }

        async function* generator(): AsyncTokenizedGenerator {
            while (true) {
                const chunk = await requestNext()
                if (!chunk) break

                for (let i = 0; i < config.batchSize; i++) {
                    const xs = []
                    const ys = []
                    for (let j = 0; j < sampleSize; j++) {
                        const idx = (i * sampleSize + j) * 2
                        const low = chunk[idx]
                        const high = chunk[idx + 1]
                        const token = toUInt16(low, high)
                        if (j < sampleSize - 1) xs.push(token)
                        if (j > 0) ys.push(token)
                    }
                    yield { xs, ys }
                }
            }
        }

        // cast as any because tf.data.generator does not take a type AsyncGenerator (but it works)
        return tf.data.generator(generator as any).map((v: any & TokenizedSample) => ({
            xs: tf.tensor1d(v.xs, 'int32'),
            ys: tf.oneHot(v.ys, vocabSize),
        }))
    }

    async loadDatasetFrom(source: string, config: TextConfig): Promise<Dataset> {
        const prefix = 'file://'
        if (!source.startsWith(prefix)) {
            source = prefix + source
        }

        let stream = this.getIteratorDatasetFromFile(source, config)
        const requestNext = async () => {
            const { value } = await stream.next()
            return value
        }
        return await this.getBackboneDataset(config, requestNext)
    }
}
