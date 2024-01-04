import { dataset, tf, Task } from '../..'
import { Model } from './model'
import { Callback, CallbackNames, Trainer } from '../trainer/trainer'

import { List } from 'immutable'

export class TFJSModel extends Model {
    static callbackNames = List.of<CallbackNames>(
        'onTrainBegin',
        'onTrainEnd',
        'onEpochBegin',
        'onEpochEnd',
        'onBatchBegin',
        'onBatchEnd'
    )

    async fit(trainer: Trainer, tuple: dataset.DataSplit): Promise<void> {
        const { training, validation } = dataset.data.data_split.extract(tuple)

        const callbacks = TFJSModel.callbackNames.reduce((map, callback) => {
            map[callback] = trainer[callback]
            return map
        }, {} as Record<CallbackNames, Callback>)

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
