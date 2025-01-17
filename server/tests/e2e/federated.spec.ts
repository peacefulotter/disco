import fs from 'fs/promises'
import path from 'node:path'
import { Server } from 'node:http'
import { Range } from 'immutable'
import { assert } from 'chai'

import { WeightsContainer, node, Disco, TrainingSchemes, client as clients, aggregator as aggregators, defaultTasks } from '@epfml/discojs-node'

import { getClient, startServer } from '../utils'

const SCHEME = TrainingSchemes.FEDERATED

describe('end-to-end federated', function () {
  this.timeout(90_000)

  let server: Server
  beforeEach(async () => {
    server = await startServer()
  })
  afterEach(() => {
    server?.close()
  })

  async function cifar10user (): Promise<WeightsContainer> {
    const dir = '../example_training_data/CIFAR10/'
    const files = (await fs.readdir(dir)).map((file) => path.join(dir, file))
    const labels = Range(0, 24).map((label) => (label % 10).toString()).toArray()

    const cifar10Task = defaultTasks.cifar10.getTask()

    const data = await new node.data.NodeImageLoader(cifar10Task).loadAll(files, { labels: labels })

    const aggregator = new aggregators.MeanAggregator(cifar10Task)
    const client = await getClient(clients.federated.FederatedClient, server, cifar10Task, aggregator)
    const disco = new Disco(cifar10Task, { scheme: SCHEME, client })

    await disco.fit(data)
    await disco.close()

    if (aggregator.model === undefined) {
      throw new Error('model was not set')
    }
    return WeightsContainer.from(aggregator.model)
  }

  async function titanicUser (): Promise<WeightsContainer> {
    const files = ['../example_training_data/titanic_train.csv']

    const titanicTask = defaultTasks.titanic.getTask()
    titanicTask.trainingInformation.epochs = 5
    const data = await (new node.data.NodeTabularLoader(titanicTask, ',').loadAll(
      files,
      {
        features: titanicTask.trainingInformation.inputColumns,
        labels: titanicTask.trainingInformation.outputColumns,
        shuffle: false
      }
    ))

    const aggregator = new aggregators.MeanAggregator(titanicTask)
    const client = await getClient(clients.federated.FederatedClient, server, titanicTask, aggregator)
    const disco = new Disco(titanicTask, { scheme: SCHEME, client, aggregator })

    await disco.fit(data)
    await disco.close()

    if (aggregator.model === undefined) {
      throw new Error('model was not set')
    }
    return WeightsContainer.from(aggregator.model)
  }

  it('two cifar10 users reach consensus', async () => {
    const [m1, m2] = await Promise.all([cifar10user(), cifar10user()])
    assert.isTrue(m1.equals(m2))
  })

  it('two titanic users reach consensus', async () => {
    const [m1, m2] = await Promise.all([titanicUser(), titanicUser()])
    assert.isTrue(
      m1.weights.some((x) => x.isNaN()) ||
      m2.weights.some((x) => x.isNaN()) ||
      m1.equals(m2)
    )
  })
})
