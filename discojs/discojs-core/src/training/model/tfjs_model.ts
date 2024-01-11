import { Task, dataset, tf } from '../..'
import { Model } from './model'
import { TrainingCallbacks, Trainer } from '../trainer/trainer'
import { train, model as gpt } from '@epfml/gpt-tfjs'

export class TFJSModel extends Model {
    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        const { training, validation } = dataset.data.data_split.extract(tuple)

        const callbacks: TrainingCallbacks = {
            onTrainBegin: trainer.onTrainBegin.bind(trainer),
            onTrainEnd: trainer.onTrainEnd.bind(trainer),
            onEpochBegin: trainer.onEpochBegin.bind(trainer),
            onEpochEnd: trainer.onEpochEnd.bind(trainer),
            onBatchBegin: trainer.onBatchBegin.bind(trainer),
            onBatchEnd: trainer.onBatchEnd.bind(trainer),
        }

        const config = (this.task as Task<gpt.GPTConfig>).trainingInformation
            .modelConfig as gpt.GPTConfig

        // FIXME + TODO: only valid for GPT-TFJS
        await train(this.model, training, config, callbacks)

        // await this.model.fitDataset(training, {
        //     epochs: this.task.trainingInformation.epochs,
        //     validationData: validation,
        //     callbacks,
        // })
    }

    toTfjs(): tf.LayersModel {
        return this.model
    }
}
