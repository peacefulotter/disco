import { describe, test, expect } from 'bun:test'
import { List, Map, Range } from 'immutable'
import fs from 'fs'

import { tf, node, Task } from '../..'

const readFilesFromDir = (dir: string): string[] =>
    fs.readdirSync(dir).map((file: string) => dir + file)

const DIRS = {
    CIFAR10: '../../example_training_data/CIFAR10/',
}

const cifar10Mock: Task = {
    id: 'cifar10',
    displayInformation: {},
    trainingInformation: {
        IMAGE_H: 32,
        IMAGE_W: 32,
        LABEL_LIST: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    },
} as unknown as Task

const mnistMock: Task = {
    id: 'mnist',
    displayInformation: {},
    trainingInformation: {
        IMAGE_H: 28,
        IMAGE_W: 28,
        LABEL_LIST: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    },
} as unknown as Task

const LOADERS = {
    CIFAR10: new node.dataset.loader.NodeImageLoader(cifar10Mock),
    MNIST: new node.dataset.loader.NodeImageLoader(mnistMock),
}
const FILES = Map(DIRS).map(readFilesFromDir).toObject()

describe('image loader', () => {
    test('loads single sample without label', async () => {
        const file = '../../example_training_data/9-mnist-example.png'
        const singletonDataset = await LOADERS.MNIST.load(file)
        const imageContent = tf.node.decodeImage(fs.readFileSync(file))
        await Promise.all(
            (
                await singletonDataset.toArrayForTest()
            ).map(async (entry) => {
                expect(await imageContent.bytes()).toEqual(
                    await (entry as tf.Tensor).bytes()
                )
            })
        )
    })

    test('loads multiple samples without labels', async () => {
        const imagesContent = FILES.CIFAR10.map((file) =>
            tf.node.decodeImage(fs.readFileSync(file))
        )
        const datasetContent = await (
            await LOADERS.CIFAR10.loadAll(FILES.CIFAR10, { shuffle: false })
        ).train.dataset.toArray()
        expect(datasetContent.length).toBe(imagesContent.length)
        expect((datasetContent[0] as tf.Tensor3D).shape).toEqual(
            imagesContent[0].shape as [number, number, number]
        )
    })

    test('loads single sample with label', async () => {
        const path = DIRS.CIFAR10 + '0.png'
        const imageContent = tf.node.decodeImage(fs.readFileSync(path))
        const datasetContent = await (
            await LOADERS.CIFAR10.load(path, { labels: ['example'] })
        ).toArray()
        expect((datasetContent[0] as any).xs.shape).toEqual(imageContent.shape)
        expect((datasetContent[0] as any).ys).toBe('example')
    })

    test('loads multiple samples with labels', async () => {
        const labels = Range(0, 24).map((label) => label % 10)
        const stringLabels = labels.map((label) => label.toString())
        const oneHotLabels = List(
            tf.oneHot(labels.toArray(), 10).arraySync() as number[]
        )

        const imagesContent = List(
            FILES.CIFAR10.map((file) =>
                tf.node.decodeImage(fs.readFileSync(file))
            )
        )
        const datasetContent = List(
            await (
                await LOADERS.CIFAR10.loadAll(FILES.CIFAR10, {
                    labels: stringLabels.toArray(),
                    shuffle: false,
                })
            ).train.dataset.toArray()
        )

        expect(datasetContent.size).toBe(imagesContent.size)
        datasetContent
            .zip(imagesContent)
            .zip(oneHotLabels)
            .forEach(([[actual, sample], label]) => {
                if (
                    !(
                        typeof actual === 'object' &&
                        actual !== null &&
                        'xs' in actual &&
                        'ys' in actual
                    )
                ) {
                    throw new Error('unexpected type')
                }
                const { xs, ys } = actual as { xs: tf.Tensor; ys: number[] }
                expect(xs.shape).toEqual(sample?.shape as any)
                expect(ys).toEqual(label as any)
            })
    })

    test('loads samples in order', async () => {
        const loader = new node.dataset.loader.NodeImageLoader(cifar10Mock)
        const dataset = await (
            await loader.loadAll(FILES.CIFAR10, { shuffle: false })
        ).train.dataset.toArray()

        List(dataset)
            .zip(List(FILES.CIFAR10))
            .forEach(async ([s, f]) => {
                const sample = (await (await loader.load(f)).toArray())[0]
                if (!tf.equal(s as tf.Tensor, sample as tf.Tensor).all()) {
                    expect(false).toBe(true)
                }
            })
        expect(true).toBe(true)
    })

    test('shuffles list', async () => {
        const loader = new node.dataset.loader.NodeImageLoader(cifar10Mock)
        const list = Range(0, 100_000).toArray()
        const shuffled = [...list]

        loader.shuffle(shuffled)
        expect(list).not.toEqual(shuffled)

        shuffled.sort((a, b) => a - b)
        expect(list).toEqual(shuffled)
    })

    test('shuffles samples', async () => {
        const loader = new node.dataset.loader.NodeImageLoader(cifar10Mock)
        const dataset = await (
            await loader.loadAll(FILES.CIFAR10, { shuffle: false })
        ).train.dataset.toArray()
        const shuffled = await (
            await loader.loadAll(FILES.CIFAR10, { shuffle: true })
        ).train.dataset.toArray()

        const misses = List(dataset)
            .zip(List(shuffled))
            .map(
                ([d, s]) =>
                    tf
                        .notEqual(d as tf.Tensor, s as tf.Tensor)
                        .any()
                        .dataSync()[0]
            )
            .reduce((acc: number, e) => acc + e)
        expect(misses).toBeGreaterThan(0)
    })
    test('validation split', async () => {
        const validationSplit = 0.2
        const imagesContent = FILES.CIFAR10.map((file) =>
            tf.node.decodeImage(fs.readFileSync(file))
        )
        const datasetContent = await new node.dataset.loader.NodeImageLoader(
            cifar10Mock
        ).loadAll(FILES.CIFAR10, {
            shuffle: false,
            validationSplit: validationSplit,
        })

        const trainSize = Math.floor(
            imagesContent.length * (1 - validationSplit)
        )
        expect((await datasetContent.train.dataset.toArray()).length).toBe(
            trainSize
        )
        if (datasetContent.validation === undefined) {
            expect(false).toBe(true)
        }
        expect(
            (await (datasetContent as any).validation.dataset.toArray()).length
        ).toEqual(imagesContent.length - trainSize)
    })
})
