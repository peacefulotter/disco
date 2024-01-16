import { dataset, tf } from '../..'
import { Model } from './model'
import { TrainingCallbacks, Trainer } from '../trainer/trainer'
import * as gpt from '../models/gpt'

export class TFJSModel<ModelConfig = unknown> extends Model<ModelConfig> {
    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        // FIXME: don't need to call dataset.data.data_split.extract(tuple)
        // because dataset does not require to be preprocessed nor batched
        // but this only works for gpt
        // TODO: add evaluate? to the task training info and don't get validation if task disables it?
        const { training, validation } = {
            training: tuple.train.dataset,
            validation: tuple.validation?.dataset ?? tuple.train.dataset,
        }

        const callbacks: TrainingCallbacks = {
            onTrainBegin: trainer.onTrainBegin.bind(trainer),
            onTrainEnd: trainer.onTrainEnd.bind(trainer),
            onEpochBegin: trainer.onEpochBegin.bind(trainer),
            onEpochEnd: trainer.onEpochEnd.bind(trainer),
            onBatchBegin: trainer.onBatchBegin.bind(trainer),
            onBatchEnd: trainer.onBatchEnd.bind(trainer),
        }

        const config = this.task.trainingInformation
            .modelConfig as gpt.GPTConfig

        // FIXME + TODO: only valid for GPT-TFJS
        await gpt.train(this.model, training, config, callbacks, validation)

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
