import { dataset, tf, training } from '../../..'
import { AdamW, clipByGlobalNormObj } from './optimizers'
import { Wandb, WandbConfig } from './wandb'
import { GPTConfig } from './model'
import evaluate from './evaluate'

export type GPTConfigWithWandb = Required<GPTConfig> & WandbConfig

const DEFAULT_CONFIG: Required<GPTConfig> = {
    lr: 0.001,
    weightDecay: 0,
    batchSize: 2,
    epochs: 9999,
    maxIter: 10_000,
    verbose: false,
    modelType: 'gpt-nano',
    evaluate: true,
    maxEvalBatches: 12,
    evaluateEvery: 100,
    blockSize: 128,
    vocabSize: 50257,
    bias: true,
    debug: false,
    dropout: 0.2,
    residDrop: 0.2,
    embdDrop: 0.2,
    nLayer: 3,
    nHead: 3,
    nEmbd: 48,
    tokEmb: true,
    lmHead: true,
}

export const getConfig = (config: GPTConfig): GPTConfigWithWandb => ({
    ...DEFAULT_CONFIG,
    ...config,
    platform:
        typeof window !== 'undefined' && typeof window.document !== 'undefined'
            ? 'browser'
            : 'node',
    gpu: 'nvidia-4070-ti',
    model: config.modelType,
    backend: tf.getBackend(),
})

const getCustomAdam = (model: any, c: Required<GPTConfig>): tf.Optimizer => {
    const includeInWeightDecay: string[] = []
    const excludeFromWeightDecay: string[] = []

    model.getNamedWeights().forEach((v: any) => {
        if (
            v.name.includes('bias') ||
            v.name.includes('normalization') ||
            v.name.includes('emb')
        ) {
            excludeFromWeightDecay.push(v.name)
        } else {
            includeInWeightDecay.push(v.name)
        }
    })
    return new AdamW({
        learningRate: c.lr,
        weightDecayRate: c.weightDecay,
        includeInWeightDecay,
        excludeFromWeightDecay,
    })
}

export async function train(
    model: tf.LayersModel,
    ds: dataset.Dataset,
    config: GPTConfig,
    callbacks: training.TrainingCallbacks,
    evalDs?: dataset.Dataset
): Promise<void> {
    const c = getConfig(config)
    console.log(c)

    const opt = c.weightDecay ? getCustomAdam(model, c) : tf.train.adam(c.lr)

    const wandb = new Wandb(c)

    callbacks.onTrainBegin()

    let epoch = 1
    let iteration = 1
    let iterator = await ds.iterator()

    const start = Date.now()
    let time = start

    while (true) {
        console.time('gpt-iter')

        callbacks.onBatchBegin(iteration)

        // Get new batch of x and y
        let next = await iterator.next()
        if (next.done) {
            callbacks.onEpochEnd(epoch)
            epoch++
            if (c.epochs && epoch > c.epochs) {
                break
            }
            callbacks.onEpochBegin(epoch)
            iterator = await ds.iterator()
            next = await iterator.next()
        }
        const { xs, ys } = next.value

        // Calculates loss, computes gradients and applies them
        const loss = tf.tidy(() => {
            let { grads, value: loss } = opt.computeGradients(() => {
                const logits = model.apply(xs)
                const loss = tf.losses.softmaxCrossEntropy(ys, logits)
                return loss as tf.Scalar
            })
            let gradsClipped = clipByGlobalNormObj(grads, 1)
            opt.applyGradients(gradsClipped)
            return loss
        })

        const lossVal = await loss.array()

        callbacks.onBatchEnd(iteration)

        // Create a WandB log payload, evaluate every
        const payload = {
            'train/perplexity': Math.exp(lossVal),
            'train/loss': lossVal,
            iter: iteration,
            'tf-mem': tf.memory().numBytes,
            dt_ms: Date.now() - time,
            time_s: (Date.now() - start) / 1000,
        }

        if (c.evaluate && iteration % c.evaluateEvery === 0) {
            if (!evalDs) {
                throw new Error(
                    'No evaluation dataset provided but config.evaluate is set'
                )
            }
            const evalPayload = await evaluate(model, evalDs, c)
            Object.assign(payload, evalPayload)
        }

        wandb.log(payload)
        time = Date.now()

        tf.dispose([loss, xs, ys])

        console.timeEnd('gpt-iter')

        // Check if we should stop
        iteration++
        if (c.maxIter && iteration > c.maxIter) {
            break
        }

        if (c.verbose) {
            console.log('Mem:', tf.memory())
            console.log(`Epoch: ${epoch}, Step: ${iteration}, Loss: ${lossVal}`)
        }

        await new Promise((resolve) => setTimeout(resolve, 1))
    }

    callbacks.onTrainEnd()
    wandb.finish()
}
