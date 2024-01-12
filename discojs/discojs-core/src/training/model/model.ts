import { tf, dataset, Task } from '../..'
import { Trainer } from '../trainer/trainer'

/**
 * Convenient interface to a TF.js model allowing for custom fit functions, while keeping
 * the model object compatible with TF.js.
 */
export abstract class Model<ModelConfig = unknown> {
    constructor(
        public readonly task: Task<ModelConfig>,
        protected readonly model: tf.LayersModel
    ) {}

    abstract fit(trainer: Trainer, data: dataset.DataSplit): Promise<void>

    /**
     * Unwraps the inner TF.js model.
     * @returns The inner TF.js model
     */
    abstract toTfjs(): tf.LayersModel
}
