import { tf } from '@epfml/discojs-node/src'
import { configModels } from './config'

export type BrowserBackendName = 'cpu' | 'webgl' | 'webgpu' | 'wasm'
export type NodeBackendName = 'cpu' | 'wasm' | 'tensorflow'
export type BackendName = BrowserBackendName | NodeBackendName

export type Models = keyof typeof configModels //, 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

export type EncodedDataset = tf.data.Dataset<tf.TensorContainer>
// export type Dataset = tf.data.Dataset<{ text: string }>

export type TokenizedSample = {
    x: number[]
    y: number[]
}

export type AsyncTokenizedGenerator = AsyncGenerator<TokenizedSample, void, unknown>

export type Callback = (model: any, loss: number, iter: number) => Promise<void> | void

export type Config = {
    debug: boolean
    verbose: boolean
    modelType: Models
    nLayer: number
    nHead: number
    nEmbd: number
    dataset: string
    batchSize: number
    blockSize: number
    lr: number
    maxIter: number
    shuffle: true | 'batch' | number
    weightDecay: number | false
    optimizer: string
    gradClip: number
    scheduler: string | null
    embdDrop: number
    residDrop: number
    bias: boolean
    numWorkers: number
    vocabSize: number
    wandbProject: string
    evalFreq: number
    evalSeqPrefix: string
    maxEvalBatches: number
    wandbName: string
}
