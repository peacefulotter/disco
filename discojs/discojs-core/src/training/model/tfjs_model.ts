import { dataset, tf } from '../..'
import { Model } from './model'
import { Callbacks, Trainer } from '../trainer/trainer'

export class TFJSModel extends Model {
    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        const { training, validation } = dataset.data.data_split.extract(tuple)

        const callbacks: Callbacks = {
            onTrainBegin: trainer.onTrainBegin.bind(trainer),
            onTrainEnd: trainer.onTrainEnd.bind(trainer),
            onEpochBegin: trainer.onEpochBegin.bind(trainer),
            onEpochEnd: trainer.onEpochEnd.bind(trainer),
            onBatchBegin: trainer.onBatchBegin.bind(trainer),
            onBatchEnd: trainer.onBatchEnd.bind(trainer),
        }

        await this.model.fitDataset(training, {
            epochs: this.task.trainingInformation.epochs,
            validationData: validation,
            callbacks,
        })
    }

    toTfjs(): tf.LayersModel {
        return this.model
    }
}
