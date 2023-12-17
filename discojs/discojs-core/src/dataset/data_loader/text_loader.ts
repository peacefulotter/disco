import { tf } from '../../'
import { Dataset } from '../dataset'
import { TextData, Data, DataSplit } from '../data'
import { DataLoader } from '.'
import { List } from 'immutable'

export interface TextConfig {
    blockSize: number
    batchSize: number
    vocabSize: number
}

export type TextSource = {
    train: string[]
    validation?: string[]
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

/**
 * Text data loader whose instantiable implementation is delegated by the platform-dependent Disco subprojects, namely,
 * @epfml/discojs-web and @epfml/discojs-node.
 */
// TODO: implement batch + shuffle + ?
// dataset.shuffle(BUFFER_SIZE)
export abstract class TextLoader extends DataLoader<TextSource, TextConfig> {
    static DEFAULT_CONFIG: Required<TextConfig> = {
        blockSize: 16,
        batchSize: 4,
        vocabSize: 50257,
    }

    abstract loadDatasetFrom(source: string, config: TextConfig): Promise<TokenizedDataset>

    async load(source: TextSource, config: TextConfig): Promise<TokenizedDataset> {
        const src = source.train[0]
        const ds = await this.loadDatasetFrom(src, config)
        return ds.batch(config.batchSize) as TokenizedDataset
    }

    async loadAll(source: TextSource, config?: Partial<TextConfig>): Promise<DataSplit> {
        const _config = Object.assign(TextLoader.DEFAULT_CONFIG, config || {})
        const split: Partial<DataSplit> = {}
        for await (const [k, files] of Object.entries(source)) {
            console.log(files)
            const datasets = await Promise.all(
                files.map(async (source) => await this.load({ train: [source] }, _config))
            )
            let dataset = List(datasets).reduce((acc: Dataset, dataset) => acc.concatenate(dataset))
            // dataset = config?.shuffle ? dataset.shuffle(BUFFER_SIZE) : dataset
            const data = await this.createData(dataset)
            ;(split as DataSplit)[k as keyof typeof split] = data
        }
        return split as DataSplit
    }

    async createData(dataset: Dataset): Promise<Data> {
        return await TextData.init(dataset, this.task)
    }
}
