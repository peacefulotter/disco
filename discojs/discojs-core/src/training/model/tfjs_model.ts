import { dataset, tf } from '../..'
import { Model } from './model'
import { TrainingCallbacks, Trainer } from '../trainer/trainer'
import * as gpt from '../models/gpt'

export class TFJSModel<ModelConfig = unknown> extends Model<ModelConfig> {
    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        // FIXME: don't need to call dataset.data.data_split.extract(tuple)
        const callbacks: TrainingCallbacks = {
            onTrainBegin: trainer.onTrainBegin.bind(trainer),
            onTrainEnd: trainer.onTrainEnd.bind(trainer),
            onEpochBegin: trainer.onEpochBegin.bind(trainer),
            onEpochEnd: trainer.onEpochEnd.bind(trainer),
            onBatchBegin: trainer.onBatchBegin.bind(trainer),
            onBatchEnd: trainer.onBatchEnd.bind(trainer),
        }

        // ============ NORMAL way of doing it ============
        // const { training, validation } = dataset.data.data_split.extract(tuple)
        // await this.model.fitDataset(training, {
        //     epochs: this.task.trainingInformation.epochs,
        //     validationData: validation,
        //     callbacks,
        // })
        // ================================================

        // because dataset does not require to be preprocessed nor batched
        // but this only works for gpt
        // TODO: add evaluate? to the task training info and don't get validation if task disables it?
        const { training, validation } = {
            training: tuple.train.dataset,
            validation: tuple.validation?.dataset ?? tuple.train.dataset,
        }

        const config = this.task.trainingInformation
            .modelConfig as gpt.GPTConfig

        console.log('TFJSModel.fit', config)

        // TODO: make it work with fitDataset instead of only using the gpt.train function
        //       the loss is NaN somehow with this
        // const history = await this.model.fitDataset(training, {
        //     batchesPerEpoch: this.task.trainingInformation.maxIterations,
        //     epochs: 1,
        //     validationData: validation,
        //     callbacks,
        // })
        // console.log(history)

        // FIXME + TODO: only valid for GPT-TFJS
        await gpt.train(this.model, training, config, callbacks, validation)
    }

    toTfjs(): tf.LayersModel {
        return this.model
    }
}
