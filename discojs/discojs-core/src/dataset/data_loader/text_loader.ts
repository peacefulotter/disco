import { tf } from '../../'
import { Dataset } from '../dataset'
import { TextData, Data, DataSplit } from '../data'
import { DataConfig, DataLoader } from '.'

export interface TextConfig extends DataConfig {
    blockSize: number
    batchSize: number // TODO: pass task to loader to have trainingInfo.batchSize
    vocabSize: number
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
// TODO: implement shuffle: dataset.shuffle(BUFFER_SIZE)
// TODO: implement other things related to dataset.Method
export abstract class TextLoader<Source, LoadDataset extends Dataset> extends DataLoader<
    Source,
    TextConfig
> {
    // Default config required to define TextConfig but leave DataConfig optional
    static DEFAULT_CONFIG: Required<Omit<TextConfig, keyof DataConfig>> & DataConfig = {
        blockSize: 16,
        batchSize: 4,
        vocabSize: 50257,
    }

    resolveConfig(config?: Partial<TextConfig>): TextConfig {
        return Object.assign({}, TextLoader.DEFAULT_CONFIG, config)
    }

    abstract load(source: Source, config: TextConfig): Promise<LoadDataset>

    abstract loadAll(source: Source | Source[], config?: Partial<TextConfig>): Promise<DataSplit>

    async createData(dataset: Dataset): Promise<Data> {
        return await TextData.init(dataset, this.task)
    }
}
