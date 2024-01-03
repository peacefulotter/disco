import { dataset, tf, Task } from '../..'
import { Model } from './model'
import { CallbackNames, Trainer } from '../trainer/trainer'

import { List, Map } from 'immutable'

export class TFJSModel extends Model {
    static callbackNames = List.of<CallbackNames>(
        'onTrainBegin',
        'onTrainEnd',
        'onEpochBegin',
        'onEpochEnd',
        'onBatchBegin',
        'onBatchEnd'
    )

    constructor(task: Task, private readonly model: tf.LayersModel) {
        super(task)
    }

    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        const { training, validation } = dataset.data.data_split.extract(tuple)

        await this.model.fitDataset(training, {
            epochs: this.task.trainingInformation.epochs,
            validationData: validation,
            callbacks: Map(
                TFJSModel.callbackNames.map((callback) => [callback, trainer[callback]])
            ).toObject(),
        })
    }

    toTfjs(): tf.LayersModel {
        return this.model
    }
}
