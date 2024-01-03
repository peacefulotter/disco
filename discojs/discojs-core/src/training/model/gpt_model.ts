import { tf, dataset, Task } from '../..'
import { Model } from './model'
import { Trainer } from '../trainer/trainer'

import { model as gpt } from '@epfml/gpt-tfjs'

// TODO: move this to an appropriate, generic and dedicated LLM file
export interface Tokenizer {
    encode: (lineToEncode: string, ...args: any[]) => number[]
    decode: (inputTokensToDecode: Iterable<number>) => string
}

type GPT = gpt.GPTModelType

// TODO: better type to avoid repetition between Task.trainingInformation and GPTConfig
export type GPTConfig = {
    epochs?: number
    maxIter?: number
    batchSize?: number
    blockSize?: number
    shuffle?: boolean | number | 'batch'
    lr?: number
    weightDecay?: boolean | number
    callbacks?: any[]
    verbose?: boolean
    bias?: boolean
    debug?: boolean
    embdDrop?: number
    nLayer?: number
    nHead?: number
    nEmbd?: number
    vocabSize?: number
    tokEmb?: boolean
    lmHead?: boolean
    modelType:
        | 'gpt2'
        | 'gpt2-medium'
        | 'gpt2-large'
        | 'gpt2-xl'
        | 'gpt-mini'
        | 'gpt-micro'
        | 'gpt-nano'
}

export class GPTModel extends Model {
    gpt: GPT

    constructor(task: Task, private readonly config: GPTConfig) {
        super(task)
        this.gpt = gpt.GPTLMHeadModel(config)
    }

    async fit(trainer: Trainer, tuple: dataset.data.DataSplit): Promise<void> {
        const { training } = dataset.data.data_split.extract(tuple)
        const { epochs, vocabSize } = this.task.trainingInformation

        await trainer.onTrainBegin()
        await this.gpt.train(training, {
            ...this.config,
            epochs,
            vocabSize,
            verbose: true,
            callbacks: [trainer.onEpochEnd],
        })
        await trainer.onTrainEnd()
    }

    toTfjs(): tf.LayersModel {
        return this.gpt.model
    }
}
