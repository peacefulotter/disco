import * as tf from '@tensorflow/tfjs'

import { Client, Task, TrainingInformant, Memory, ModelType } from '../..'

import { DistributedTrainer } from './distributed_trainer'
import { LocalTrainer } from './local_trainer'
import { Trainer } from './trainer'

/**
 * A class that helps build the Trainer and auxiliary classes.
 */
export class TrainerBuilder {
  constructor (
    private readonly memory: Memory,
    private readonly task: Task,
    private readonly trainingInformant: TrainingInformant
  ) {}

  /**
   * Builds a trainer object.
   *
   * @param client client to share weights with (either distributed or federated)
   * @param distributed whether to build a distributed or local trainer
   * @returns
   */
  async build (client: Client, distributed: boolean = false): Promise<Trainer> {
    const model = await this.getModel()
    if (distributed) {
      return new DistributedTrainer(
        this.task,
        this.trainingInformant,
        this.memory,
        model,
        client
      )
    } else {
      return new LocalTrainer(
        this.task,
        this.trainingInformant,
        this.memory,
        model
      )
    }
  }

  private async getModel (): Promise<tf.LayersModel> {
    const model = await this.getModelFromMemory()
    return await this.updateModelInformation(model)
  }

  /**
   *
   * @returns
   */
  private async getModelFromMemory (): Promise<tf.LayersModel> {
    const modelID = this.task.trainingInformation?.modelID
    if (modelID === undefined) {
      throw new Error('undefined model ID')
    }

    const modelExistsInMemory = await this.memory.getModelMetadata(
      ModelType.WORKING,
      this.task.taskID,
      modelID
    )
    if (modelExistsInMemory !== undefined) {
      return await this.memory.getModel(
        ModelType.WORKING,
        this.task.taskID,
        modelID
      )
    }

    throw new Error('no model in memory')
  }

  private async updateModelInformation (model: tf.LayersModel): Promise<tf.LayersModel> {
    // Continue local training from previous epoch checkpoint
    if (model.getUserDefinedMetadata() === undefined) {
      model.setUserDefinedMetadata({ epoch: 0 })
    }

    const info = this.task.trainingInformation
    if (info === undefined) {
      throw new Error('undefined training information')
    }

    model.compile(info.modelCompileData)

    if (info.learningRate !== undefined) {
      // TODO: Not the right way to change learningRate and hence we cast to any
      // the right way is to construct the optimiser and pass learningRate via
      // argument.
      (model.optimizer as any).learningRate = info.learningRate
    }

    return model
  }
}