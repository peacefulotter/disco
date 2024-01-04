import { dataset, tf } from '../..'
import { Model } from './model'
import { Callbacks, Trainer } from '../trainer/trainer'

export class TFJSModel extends Model {
    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        const { training, validation } = dataset.data.data_split.extract(tuple)

        const callbacks: Callbacks = {
            onTrainBegin: trainer.onTrainBegin,
            onTrainEnd: trainer.onTrainEnd,
            onEpochBegin: trainer.onEpochBegin,
            onEpochEnd: trainer.onEpochEnd,
            onBatchBegin: trainer.onBatchBegin,
            onBatchEnd: trainer.onBatchEnd,
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
