import { Task, tf } from '../../'
import { Dataset } from '../dataset'
import { TextData, Data, DataSplit } from '../data'
import { DataConfig, DataLoader } from '.'

export interface TextConfig extends DataConfig {
    blockSize: number
    vocabSize: number
}

type TokenizedSample = {
    xs: number[]
    ys: number[]
}

export type TokenizedTensorSample = {
    xs: tf.Tensor1D // tokens of size (blockSize)
    ys: tf.Tensor2D // one hot encoded vector of size (blockSize, vocabSize)
}

export type BatchedTokenizedTensorSample = {
    xs: tf.Tensor2D // tokens of size (B, blockSize)
    ys: tf.Tensor3D // one hot encoded vector of size (B, blockSize, vocabSize)
}

export type TokenizedDataset = Dataset<TokenizedTensorSample>

export type TokenizedIterResult = IteratorResult<
    BatchedTokenizedTensorSample,
    BatchedTokenizedTensorSample
>

export type TextSource = {
    train: string[]
    validation?: string[]
}

export type ParsedWSSearchParams = {
    config: TextConfig
    file: string
    task: Task
}
export type WSSearchParams = Record<keyof ParsedWSSearchParams, string>

type AsyncTokenizedGenerator = AsyncGenerator<TokenizedSample, void, unknown>

/**
 * Text data loader whose instantiable implementation is delegated by the platform-dependent Disco subprojects, namely,
 * @epfml/discojs-web and @epfml/discojs-node.
 */
// TODO: implement shuffle: dataset.shuffle(BUFFER_SIZE)
export abstract class TextLoader extends DataLoader<
    string,
    TextSource,
    TextConfig
> {
    // Default config required to define TextConfig but leave DataConfig optional
    static DEFAULT_CONFIG: Required<Omit<TextConfig, keyof DataConfig>> &
        DataConfig = {
        blockSize: 16,
        vocabSize: 50257,
    }

    batchSize: number

    constructor(protected task: Task) {
        super(task)
        this.batchSize = this.task.trainingInformation.batchSize
        if (!this.batchSize)
            throw new Error(
                'batch size is undefined, define a batchSize in your task training information (task.trainingInformation.batchSize)'
            )
    }

    resolveConfig(config?: Partial<TextConfig>): TextConfig {
        return Object.assign({}, TextLoader.DEFAULT_CONFIG, config)
    }

    /**
     * Core dataset, shared between node and web versions
     * Takes an iterator that yields arrays of numbers and turns
     * them into structured batch of tuples x, y
     * @param config
     * @param requestNext
     * @returns A TokenizedDataset = tfjs dataset containing xs and ys tensors
     */
    async getCoreDataset(
        config: TextConfig,
        iterator: AsyncIterator<Buffer, Buffer, Buffer>
    ): Promise<TokenizedDataset> {
        const { vocabSize } = config
        const sampleSize = config.blockSize + 1

        const toUInt16 = (low: number, high: number) => {
            low &= 0xff
            high &= 0xff
            return (high << 8) | low
        }

        const batchSize = this.batchSize

        async function* generator(): AsyncTokenizedGenerator {
            let next = iterator.next()
            while (true) {
                const { value: chunk } = await next
                // const chunk = value.toJSON().data
                if (!chunk) break

                // pre-fetch the next chunk even before actually requesting it
                next = iterator.next()

                // console.time('generator')
                for (let i = 0; i < batchSize; i++) {
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

                    // console.time('next')
                    yield { xs, ys }
                    // console.timeEnd('next')
                }
                // console.timeEnd('generator')
            }
        }

        // cast as any because tf.data.generator does not take a type AsyncGenerator (but it works)
        return tf.data
            .generator(generator as any)
            .map((v: any & TokenizedSample) => ({
                xs: tf.tensor1d(v.xs, 'int32'),
                ys: tf.oneHot(v.ys, vocabSize),
            })) as TokenizedDataset
    }

    abstract load(source: string, config: TextConfig): Promise<TokenizedDataset>

    abstract loadAll(
        source: TextSource,
        config?: Partial<TextConfig>
    ): Promise<DataSplit>

    async createData(dataset: Dataset): Promise<Data> {
        return await TextData.init(dataset, this.task)
    }
}
