import { List } from 'immutable'
import { tf, node, Task } from '../..'
import { describe, test, expect } from 'bun:test'

const inputFiles = ['../../example_training_data/titanic_train.csv']

const titanicMock: Task = {
    taskID: 'titanic',
    displayInformation: {},
    trainingInformation: {
        inputColumns: [
            'PassengerId',
            'Age',
            'SibSp',
            'Parch',
            'Fare',
            'Pclass',
        ],
        outputColumns: ['Survived'],
    },
} as unknown as Task

describe('tabular loader', () => {
    test('loads a single sample', async () => {
        const loaded = new node.dataset.loader.NodeTabularLoader(
            titanicMock,
            ','
        ).loadAll(inputFiles, {
            features: titanicMock.trainingInformation?.inputColumns,
            labels: titanicMock.trainingInformation?.outputColumns,
            shuffle: false,
        })
        const sample = await (
            await (await loaded).train.dataset.iterator()
        ).next()
        /**
         * Data loaders simply return a dataset object read from input sources.
         * They do NOT apply any transform/conversion, which is left to the
         * dataset builder.
         */
        expect(sample).toEqual({
            value: {
                xs: [1, 3, 22, 1, 0, 7.25],
                ys: [0],
            },
            done: false,
        })
    })

    test('shuffles samples', async () => {
        const titanic = titanicMock
        const loader = new node.dataset.loader.NodeTabularLoader(titanic, ',')
        const config = {
            features: titanic.trainingInformation?.inputColumns,
            labels: titanic.trainingInformation?.outputColumns,
            shuffle: false,
        }
        const dataset = await (
            await loader.loadAll(inputFiles, config)
        ).train.dataset.toArray()
        const shuffled = await (
            await loader.loadAll(inputFiles, { ...config, shuffle: true })
        ).train.dataset.toArray()

        const misses = List(dataset)
            .zip(List(shuffled))
            .map(
                ([d, s]) =>
                    tf
                        .notEqual((d as any).xs, (s as any).xs)
                        .any()
                        .dataSync()[0]
            )
            .reduce((acc: number, e) => acc + e)
        expect(misses).toBeGreaterThan(0)
    })
})
