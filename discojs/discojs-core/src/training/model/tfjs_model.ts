import { data, tf, Task } from '../..'
import { Model } from './model'
import { Trainer } from '../trainer/trainer'

export class TFJSModel extends Model {
  constructor (
    task: Task,
    private readonly model: tf.LayersModel
  ) {
    super(task)
  }

  async fit (trainer: Trainer, tuple: data.tuple.DataSplit): Promise<void> {
    const { training, validation } = data.tuple.extract(tuple)

    await this.model.fitDataset(training, {
      epochs: this.task.trainingInformation.epochs,
      validationData: validation,
      callbacks: {
        onTrainBegin: trainer.onTrainBegin,
        onTrainEnd: trainer.onTrainEnd,
        onEpochBegin: trainer.onEpochBegin,
        onEpochEnd: trainer.onEpochEnd,
        onBatchBegin: trainer.onBatchBegin,
        onBatchEnd: trainer.onBatchEnd
      }
    })
  }

  toTfjs (): tf.LayersModel {
    return this.model
  }
}