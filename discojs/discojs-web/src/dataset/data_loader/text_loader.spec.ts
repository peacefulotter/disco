/// <reference lib="dom" />
import { GlobalRegistrator } from '@happy-dom/global-registrator'
const oldConsole = console
GlobalRegistrator.register()
window.console = oldConsole

import fs from 'fs'
import path from 'path'
/* @ts-ignore */
import { describe, test, expect } from 'bun:test'
import { encode, decode } from 'gpt-tokenizer/esm/model/text-davinci-003'
import * as disco from '../..'

const task = disco.defaultTasks.wikitext.getTask()
const config = {
    ...task.trainingInformation.modelConfig,
    blockSize: 3,
    batchSize: 2,
    vocabSize: 50257,
}

const datasetsFolder = path.join('../../experiment', 'datasets', 'wikitext-103')
const source: disco.dataset.TextSource = {
    train: [path.join(datasetsFolder, 'test.tokens')],
    validation: [path.join(datasetsFolder, 'validation.tokens')],
}

const getIterator = async () => {
    const loaded = await new disco.browser.dataset.loader.WebTextLoader(
        task
    ).loadAll(source, config)
    const ds = loaded.train.dataset as disco.dataset.TokenizedDataset
    const iter = await ds.batch(config.batchSize).iterator()
    return {
        next: async () => {
            const { value } =
                (await iter.next()) as disco.dataset.TokenizedIterResult
            const { xs, ys } = value
            const x = xs.flatten()
            const y = (ys.argMax(2) as disco.tf.Tensor2D).flatten() // get indices of max values along last axis
            return { xs, ys, x, y }
        },
    }
}

/**
 * Reads the RAW dataset (not preprocessed) and tokenizes the equivalent of the first batch.
 */
const getRawTokenizedSample = async () => {
    const size = config.batchSize * (config.blockSize + 1)
    const wikiRaw = fs.createReadStream(path.join(datasetsFolder, 'test'), {
        encoding: 'utf8',
        start: 0,
        end: size * 3, // * 3 to account for spaces between words and tabs
    })
    const iter = wikiRaw.iterator()
    const { value: chunk } = await iter.next()
    const tokens = encode(chunk).slice(0, size)
    return tokens
}

describe('web text loader', () => {
    test('loads a batched sample', async () => {
        const iter = await getIterator()
        const { xs, ys, x, y } = await iter.next()

        expect(xs.shape).toEqual([config.batchSize, config.blockSize])
        expect(ys.shape).toEqual([
            config.batchSize,
            config.blockSize,
            config.vocabSize,
        ])
        expect(x.equal([220, 198, 796, 347, 2852, 353]))
        expect(y.equal(disco.tf.tensor([198, 796, 5199, 2852, 353, 796])))

        disco.tf.dispose([xs, ys, x, y])
    })

    test('iterates properly', async () => {
        const iter = await getIterator()
        for (let i = 0; i < 10; i++) {
            const { xs, ys, x, y } = await iter.next()
            expect(xs.shape).toEqual([config.batchSize, config.blockSize])
            expect(ys.shape).toEqual([
                config.batchSize,
                config.blockSize,
                config.vocabSize,
            ])
            const x_arr = await x.array()
            const y_arr = await y.array()
            console.log(x_arr, y_arr)
            console.log('x=', decode(x_arr).trim())
            console.log('y=', decode(y_arr).trim())
            disco.tf.dispose([xs, ys, x, y])
        }
    })

    test('dataset is tokenized properly', async () => {
        const tokens = await getRawTokenizedSample()
        const iter = await getIterator()
        const { xs, ys, x, y } = await iter.next()

        /**
         * Flatten the batch by taking the first token in x and the rest in y, since y is x shifted by 1 + 1 token
         * e.g. [a, b, c, d, e, f] -> x = [a, b, c, d, e] and y = [b, c, d, e, f]
         * thus x[0] + y = [a, b, c, d, e, f]
         **/
        const xs_arr = await xs.array()
        const ys_arr = await (ys.argMax(2) as disco.tf.Tensor2D).array() // get indices of max values along last axis
        const arr: number[] = []
        for (let i = 0; i < config.batchSize; i++) {
            arr.push(xs_arr[i][0], ...ys_arr[i])
        }

        expect(arr).toEqual(tokens)

        disco.tf.dispose([xs, ys, x, y])
    })

    test('benchmark 2.000 iterations', async () => {
        const iterations = 2_000
        const iter = await getIterator()
        const benchmarkStart = Date.now()
        for (let i = 0; i < iterations; i++) {
            const { xs, ys, x, y } = await iter.next()
            disco.tf.dispose([xs, ys, x, y])
        }
        const benchmarkEnd = Date.now()
        const ms = benchmarkEnd - benchmarkStart
        const duration = ms / 1000
        console.log(
            `Time taken: ${duration}s, time per iteration: ${(
                ms / iterations
            ).toFixed(3)}ms`
        )
    })
})
