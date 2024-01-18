import {
    tf,
    training,
    Memory,
    Task,
    TrainingInformant,
    WeightsContainer,
    client as clients,
} from '../..'
import { Aggregator } from '../../aggregator'
import { Trainer } from './trainer'

/**
 * Class whose role is to train a model in a distributed way with a given dataset.
 */
export class DistributedTrainer extends Trainer {
    private readonly aggregator: Aggregator

    /**
     * DistributedTrainer constructor, accepts same arguments as Trainer and in additional also a client who takes care of communicating weights.
     */
    constructor(
        task: Task,
        trainingInformant: TrainingInformant,
        memory: Memory,
        model: training.model.Model,
        private readonly client: clients.Client
    ) {
        super(task, trainingInformant, memory, model)
        this.aggregator = this.client.aggregator
        this.aggregator.setModel(model)
    }

    async onTrainBegin(logs?: tf.Logs): Promise<void> {
        console.log('[DEC TRAINER] ON TRAIN BEGIN', logs)

        await super.onTrainBegin(logs)

        const weights = WeightsContainer.from(this.model)

        await this.client.onTrainBeginCommunication(
            weights,
            this.trainingInformant
        )
        console.log('[DEC TRAINER] ON TRAIN BEGIN DONE')
    }

    async onRoundBegin(accuracy: number): Promise<void> {
        console.log('[DEC TRAINER] ON ROUND BEGIN', accuracy)

        const weights = WeightsContainer.from(this.model)

        await this.client.onRoundBeginCommunication(
            weights,
            this.roundTracker.round,
            this.trainingInformant
        )

        console.log('[DEC TRAINER] ON ROUND BEGIN DONE')
    }

    /**
     * Callback called every time a round is over
     */
    async onRoundEnd(accuracy: number): Promise<void> {
        console.log('[DEC TRAINER] ON ROUND END')

        const weights = WeightsContainer.from(this.model)

        console.log('[DEC TRAINER] ON ROUND END 1')

        await this.client.onRoundEndCommunication(
            weights,
            this.roundTracker.round,
            this.trainingInformant
        )

        console.log('[DEC TRAINER] ON ROUND END 2')
        if (this.aggregator.model !== undefined) {
            // The aggregator's own aggregation is async. The trainer updates its model to match the aggregator's
            // after it has completed a round of training.
            console.log('[DEC TRAINER] ON ROUND END 3')
            this.model.tfjs.setWeights(this.aggregator.model.tfjs.getWeights())
            console.log('[DEC TRAINER] ON ROUND END 4')
        }

        console.log('[DEC TRAINER] ON ROUND END 5')

        await this.memory.updateWorkingModel(
            {
                taskID: this.task.id,
                name: this.task.trainingInformation.modelID,
            },
            this.model.tfjs
        )
        console.log('[DEC TRAINER] ON ROUND END DONE')
    }
}
