import { tf, dataset, Task } from '../..'
import { Model } from './model'
import { Trainer } from '../trainer/trainer'

// import { model as gpt } from '@epfml/gpt-tfjs'

// TODO: move this to an appropriate, generic and dedicated LLM file
export interface Tokenizer {
    encode: (lineToEncode: string, ...args: any[]) => number[]
    decode: (inputTokensToDecode: Iterable<number>) => string
}

// type GPT = gpt.GPTModelType

// export class GPTModel extends Model {
//     constructor(task: Task, private readonly gpt: GPT) {
//         super(task, gpt.model)
//     }

//     async fit(trainer: Trainer, tuple: dataset.data.DataSplit): Promise<void> {
//         const { training } = dataset.data.data_split.extract(tuple)
//         const { epochs, vocabSize } = this.task.trainingInformation

//         await trainer.onTrainBegin()
//         await this.gpt.train(training, {
//             ...this.config,
//             epochs,
//             vocabSize,
//             verbose: true,
//             callbacks: [trainer.onEpochEnd],
//         })
//         await trainer.onTrainEnd()
//     }

//     toTfjs(): tf.LayersModel {
//         return this.gpt.model
//     }
// }
